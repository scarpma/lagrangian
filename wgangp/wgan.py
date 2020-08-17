#!/usr/bin/env python
# coding: utf-8

from db_utils import *
from gen import *
from critic import *


class RandomWeightedAverage(tensorflow.keras.layers.Layer):
    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size

    def call(self, inputs, **kwargs):
        alpha = tensorflow.random_uniform((self.batch_size, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

    def compute_output_shape(self, input_shape):
        return input_shape[0]



class WGANGP():
    def __init__(self,gen,critic,noise_dim,n_critic,batch_size,text):
        #self.d_losses = []
        #self.d_losses_test = []
        #self.g_losses = []
        self.real_critics = []
        self.fake_critics = []
        self.sig_len = 2000
        self.channels = 1
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.text = text

        # Creo cartella della run attuale:
        if not self.text == '0':
            self.current_run = self.get_run()
            self.dir_path = f"runs/{self.current_run}/"
            os.mkdir(self.dir_path)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = n_critic
        #self.gen_lr = self.critic_lr = 0.00001
        self.gen_lr = 0.00005
        self.critic_lr = 0.0001
        #self.gen_lr = 0.00000001
        #self.critic_lr = 0.000001
        self.gen_b1 = self.critic_b1 = 0.0 # di solito è 0.0
        self.gen_b2 = self.critic_b2 = 0.9 # di solito è 0.9
        gen_optimizer = Adam(learning_rate=self.gen_lr, beta_1=self.gen_b1,beta_2=self.gen_b2)
        critic_optimizer = Adam(learning_rate=self.critic_lr, beta_1=self.critic_b1,beta_2=self.critic_b2)
        

        self.critic = critic
        self.gen = gen


        #-------------------------------
        # Construct Computational Graph
        #       for the Critic
        #-------------------------------

        # Freeze generator's layers while training critic
        self.gen.trainable = False

        # Image input (real sample)
        real_img = Input(shape=(self.sig_len,self.channels))

        # Noise input
        z_disc = Input(shape=(self.noise_dim,))
        # Generate image based of noise (fake sample)
        fake_img = self.gen(z_disc)

        # Discriminator determines validity of the real and fake images
        fake = self.critic(fake_img)
        valid = self.critic(real_img)

        # Construct weighted average between real and fake images
        interpolated_img = RandomWeightedAverage(self.batch_size)([real_img, fake_img])
        # Determine validity of weighted sample
        validity_interpolated = self.critic(interpolated_img)

        # Use Python partial to provide loss function with additional
        # 'averaged_samples' argument
        partial_gp_loss = partial(self.gradient_penalty_loss,
                                  averaged_samples=interpolated_img)
        partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

        self.critic_model = Model(inputs=[real_img, z_disc],
                                  outputs=[valid, fake, validity_interpolated])
        self.critic_model.compile(loss=[self.wasserstein_loss,
                                        self.wasserstein_loss,
                                        partial_gp_loss],
                                  optimizer=critic_optimizer,
                                  loss_weights=[1, 1, 10])



        #-------------------------------
        # Construct Computational Graph
        #         for Generator
        #-------------------------------

        # For the generator we freeze the critic's layers
        self.critic.trainable = False
        self.gen.trainable = True

        # Sampled noise for input to generator
        z_gen = Input(shape=(self.noise_dim,))
        # Generate images based of noise
        img = self.gen(z_gen)
        # Discriminator determines validity
        valid = self.critic(img)
        # Defines generator model
        self.gen_model = Model(z_gen, valid)
        self.gen_model.compile(loss=self.wasserstein_loss, optimizer=gen_optimizer)


        
    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on prediction and weighted real / fake samples
        """
        gradients = K.gradients(y_pred, averaged_samples)[0]
        # compute the euclidean norm by squaring ...
        gradients_sqr = K.square(gradients)
        #   ... summing over the rows ...
        gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
        #   ... and sqrt
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        # compute lambda * (1 - ||grad||)^2 still for each single sample
        gradient_penalty = K.square(1 - gradient_l2_norm)
        # return the mean as loss over all the batch samples
        return K.mean(gradient_penalty)

    
    def get_run(self):
        runs = glob.glob('runs/*/')
        runs = sorted(runs, key=lambda x: int(x[5:-1]))
        # print('RUNS = ',runs)
        runs = [int(line[5:-1]) for line in runs]
        
        if len(runs)>0: return runs[-1] + 1
        else: return 1

 
    def plot_trajs(self, gen_trajs,epoch):
        plt.figure(figsize=(15, 2*len(gen_trajs)))
        for i, traj in enumerate(gen_trajs):
            plt.subplot(len(gen_trajs), 1, i+1)
            plt.plot(traj)
        plt.tight_layout()
        plt.savefig(self.dir_path+f'{epoch}_gen_traj.png', fmt='png', dpi=100)
        plt.close()
   
 
    def train(self, gen_iters, db_train, db_test):
        
        # salvo info log #
        with open(self.dir_path+'logs.txt','a+') as f:
            f.write(f"TRAINING\nbatch={self.batch_size}\nncritic={self.n_critic}\ngen_iterations={gen_iters}\ngen_lr={self.gen_lr} gen_b1={self.gen_b1}  gen_b2={self.gen_b2}\ncritic_lr={self.critic_lr} critic_b1={self.critic_b1} critic_b2={self.critic_b2}\n")
            f.write(self.text)
        fl = open(self.dir_path+'training.dat','a+')
        fl.write(f"# gen_iteration, critic_iteration, d_loss_tot, d_loss_true, d_loss_fake, d_loss_gp, d_loss_tot_test, d_loss_true_test, d_loss_fake_test, d_loss_gp_test, g_loss\n")
        fl.close()

        # ############## #
        
        static_noise = np.random.normal(0, 1, size=(1, self.noise_dim))
        d_loss_test = [0., 0., 0., 0.]
        g_loss = 0.
        
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty

        print(f'\nNCRITIC = {self.n_critic}\n')
        

        for gen_iter in range(gen_iters):            

            fl = open(self.dir_path+'training.dat','a+')
            for jj in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, db_train.shape[0], self.batch_size)
                # select a precise batch of images
                # batch_start = (gen_iter * self.n_critic + jj) * self.batch_size
                # batch_end = (gen_iter * self.n_critic + jj + 1) * self.batch_size
                # idx = np.arange(batch_start, batch_end)
                imgs = db_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                #noise = np.random.uniform(-1., 1., (self.batch_size, self.noise_dim))
                # Train the critic
                # print("init disc train") 
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                           [valid, fake, dummy])
                # print("disc trained")
                if (jj!=self.n_critic-1): fl.write("%7d %7d %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n"%(gen_iter,gen_iter*self.n_critic+jj,d_loss[0],d_loss[1],d_loss[2],d_loss[3],d_loss_test[0],d_loss_test[1],d_loss_test[2],d_loss_test[3],g_loss))


            # ---------------------
            #  Train Generator
            # ---------------------

            idx = np.random.randint(0, db_test.shape[0], self.batch_size)
            imgs = db_test[idx]
            #d_loss_test = self.critic_model.test_on_batch([imgs, noise],
            #                                              [valid, fake, dummy])
            # print("init gen train") 
            g_loss = self.gen_model.train_on_batch(noise, valid)
            # print("gen trained")

            # Plot the progress
            print(f"Gen_Iter: {gen_iter:6d} [D loss: {d_loss[0]:9.2e}] [d_loss_test: {d_loss_test[0]:9.2e}] [G loss: {g_loss:9.2e}]")
            fl.write("%7d %7d %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e %11.4e\n"%(gen_iter,gen_iter*self.n_critic+jj,d_loss[0],d_loss[1],d_loss[2],d_loss[3],d_loss_test[0],d_loss_test[1],d_loss_test[2],d_loss_test[3],g_loss))
            fl.close()

            # If at save interval => save generated image samples
            if gen_iter % 100 == 0:
                self.plot_trajs(self.gen.predict(np.random.normal(0,1, size=(3,self.noise_dim))), gen_iter)
            if gen_iter % 250 == 0:    
                self.critic.save(self.dir_path+f'{gen_iter}_critic.h5')
                self.gen.save(self.dir_path+f'{gen_iter}_gen.h5')
            if gen_iter % 2000 == 0:# and gen_iter > 0:    
                mini_db = self.gen.predict(np.random.normal(0, 1, size=(50000, self.noise_dim)))
                np.save(self.dir_path+f'gen_trajs_{gen_iter}', mini_db)
                del mini_db


        self.critic.save(self.dir_path+f'{gen_iter+1}_critic.h5')
        self.gen.save(self.dir_path+f'{gen_iter+1}_gen.h5')
        

            
if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_iters', type=int, default=20000)
    parser.add_argument('--ncritic', type=int, default=5)
    parser.add_argument('--load', type=int, default=[0], nargs=2)
    args = parser.parse_args()
    gen_iters = args.gen_iters
    load = args.load
    ncritic = args.ncritic
    
    print("Importing Databases ... ",end="")
    db_train, db_test = load_data(0.8)
    print("Done")
    
    noise_dim = 100
    
    if load[0] > 0:
        run = load[0]
        number = load[1]
        path_gen = f'runs/{run}/{number}_gen.h5'
        path_critic = f'runs/{run}/{number}_critic.h5'
        #def wasserstein_loss(y_true, y_pred):
        #    return K.mean(y_true * y_pred)
        #tensorflow.keras.losses.wasserstein_loss = wasserstein_loss
        #gen = load_model(path_gen, custom_objects={'wasserstein_loss': wasserstein_loss})
        gen = load_model(path_gen)
        #critic = load_model(path_critic, custom_objects={'wasserstein_loss': wasserstein_loss})
        critic = load_model(path_critic)
        critic.trainable = True
        
        # scrivo stringa info log gen #
        text = f"continuo run {run} from number {number}"
    else:
        fs=(100,1)
        fm=128
        init_sigma = 0.003
        init_mean = 0.0
        alpha = 0.2
        # scrivo stringa info log gen #
        text = f"GEN\n{fs,fm,init_sigma,init_mean,noise_dim,alpha}\n"
        gen = build_generator(fs,fm,init_sigma,init_mean,alpha,noise_dim)
        fs = 100
        fm = 128
        init_sigma = 0.02
        init_mean = 0.0
        alpha = 0.2
        critic = build_critic(fs,fm,init_sigma,init_mean,alpha)
        # scrivo stringa info log critic #
        text += f"CRITIC\n{fs,fm,init_sigma,init_mean,alpha}\n"

            
            
            
    
    wgan = WGANGP(gen, critic, noise_dim, ncritic, 500, text)
    print(f'Train for {gen_iters} generator iterations')
    wgan.train(gen_iters, db_train, db_test)
