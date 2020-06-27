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
        # optimizer = RMSprop(lr=0.00005)
        optimizer = Adam(learning_rate=0.0001, beta_1=0.5,beta_2=0.9)
        # optimizer = Adam(0.00005)
        # optimizer = Adam(0.000005)


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
                                  optimizer=optimizer,
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
        self.gen_model.compile(loss=self.wasserstein_loss, optimizer=optimizer)


        
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

    def plot_disc_predictions(self, fake_disc, real_disc, epoch, batch_size=250):
        plt.figure()
        plt.plot(real_disc, label='real')
        plt.plot(fake_disc, label='fake')
        plt.legend()
        plt.savefig(self.dir_path+f'{epoch}_critic_pred.png', fmt='png', dpi=100)
        plt.close()
    
    def train(self, epochs, db_train, db_test):
        
        # salvo info log #
        with open(self.dir_path+'logs.txt','a+') as f:
            f.write(f"TRAINING\nncritic={self.n_critic}\nepochs={epochs}\n")
            f.write(self.text)
        fl = open(self.dir_path+'training.dat','a+')
        fl.write(f"# iter, d_loss_tot, d_loss_true, d_loss_fake, d_loss_gp, d_loss_tot_test, d_loss_true_test, d_loss_fake_test, d_loss_gp_test, g_loss\n")

        # ############## #
        
        static_noise = np.random.normal(0, 1, size=(1, self.noise_dim))
        d_loss_test = [0., 0., 0., 0.]
        g_loss = 0.
        
        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty
        
        print(f'\nNCRITIC = {self.n_critic}\n')
 
        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, db_train.shape[0], self.batch_size)
                imgs = db_train[idx]
                # Sample generator input
                noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                #noise = np.random.uniform(-1., 1., (self.batch_size, self.noise_dim))
                # Train the critic
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                           [valid, fake, dummy])
                #print(d_loss)
                fl.write("%7d %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g\n"%(epoch,d_loss[0],d_loss[1],d_loss[2],d_loss[3],d_loss_test[0],d_loss_test[1],d_loss_test[2],d_loss_test[3],g_loss))

            # ---------------------
            #  Train Generator
            # ---------------------

            idx = np.random.randint(0, db_test.shape[0], self.batch_size)
            imgs = db_test[idx]
            d_loss_test = self.critic_model.test_on_batch([imgs, noise],
                                                          [valid, fake, dummy])
            g_loss = self.gen_model.train_on_batch(noise, valid)
            #self.g_losses.append(g_loss)
            #self.d_losses.append(d_loss)
            #self.d_losses_test.append(d_loss_test)

            # Plot the progress
            print(f"{epoch} [D loss: {d_loss[0]:6.2g}] [d_loss_test: {d_loss_test[0]:6.2g}] [G loss: {g_loss:6.2g}]")
            fl.write("%7d %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g %13.7g\n"%(epoch,d_loss[0],d_loss[1],d_loss[2],d_loss[3],d_loss_test[0],d_loss_test[1],d_loss_test[2],d_loss_test[3],g_loss))

            # If at save interval => save generated image samples
            if epoch % 100 == 0:
                self.plot_trajs(self.gen.predict(np.random.normal(0,1, size=(3,self.noise_dim))), epoch)
                #self.plot_disc_predictions(fake_critic, real_critic, epoch, batch_size)
            if epoch % 250 == 0:    
                self.critic.save(self.dir_path+f'{epoch}_critic.h5')
                self.gen.save(self.dir_path+f'{epoch}_gen.h5')


        fl.close()
        self.critic.save(self.dir_path+f'{epochs-1}_critic.h5')
        self.gen.save(self.dir_path+f'{epochs-1}_gen.h5')
        

            
if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=6000)
    parser.add_argument('--ncritic', type=int, default=30)
    parser.add_argument('--load', type=int, default=[0], nargs=2)
    args = parser.parse_args()
    epochs = args.epochs
    load = args.load
    ncritic = args.ncritic
    
    db_train, db_test = load_data(0.8)
    
    noise_dim = 100
    
    if load[0] > 0:
        run = load[0]
        number = load[1]
        path_gen = f'runs/{run}/{number}_gen.h5'
        path_critic = f'runs/{run}/{number}_critic.h5'
        def wasserstein_loss(y_true, y_pred):
            return K.mean(y_true * y_pred)
        tensorflow.keras.losses.wasserstein_loss = wasserstein_loss
        gen = load_model(path_gen, custom_objects={'wasserstein_loss': wasserstein_loss})
        critic = load_model(path_critic, custom_objects={'wasserstein_loss': wasserstein_loss})
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
    print(f'train for {epochs} epochs')
    wgan.train(epochs, db_train, db_test)
