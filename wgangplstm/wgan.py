#!/usr/bin/env python
# coding: utf-8


# IMPORT MODULES

from db_utils import *


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

    def __init__(self,gen,critic,noise_dim,n_critic,batch_size,gen_lr,critic_lr,text):
        #self.d_losses = []
        #self.d_losses_test = []
        #self.g_losses = []
        self.real_critics = []
        self.fake_critics = []
        self.sig_len = 2048 #VAR
        self.channels = 1 #VAR
        self.noise_dim = noise_dim
        self.batch_size = batch_size
        self.text = text

        # Creo cartella della run attuale:
        if not self.text == '0':
            self.current_run = self.get_run()
            self.dir_path = f"runs/{self.current_run}/"
            print(f"\nCreating new directories for run: {self.current_run}\n")
            os.mkdir(self.dir_path)

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = n_critic
        #self.gen_lr = self.critic_lr = 0.00001
        self.gen_lr = gen_lr
        self.critic_lr = critic_lr
        self.gen_b1 = self.critic_b1 = 0.0 # wgangp article: 0.0
        self.gen_b2 = self.critic_b2 = 0.9 # wgangp article: 0.9
        gen_optimizer = Adam(learning_rate=self.gen_lr,
                             beta_1=self.gen_b1,beta_2=self.gen_b2)
        critic_optimizer = Adam(learning_rate=self.critic_lr,
                                beta_1=self.critic_b1,beta_2=self.critic_b2)

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
        interpolated_img = RandomWeightedAverage(
                self.batch_size)([real_img, fake_img])
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
        print(self.critic_model.summary())


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
        print(self.gen_model.summary())


    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)


    def gradient_penalty_loss(self, y_true, y_pred, averaged_samples):
        """
        Computes gradient penalty based on
        prediction and weighted real / fake samples
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
        longest_string = max(runs, key=len)

        print(longest_string)
        if len(longest_string) > 8 :
            print("Attention: more than 99 runs are not supported. Exiting.")
            exit()

        runs = sorted(runs, key=lambda x: int(x[5:-1]))
        runs = [int(line[5:-1]) for line in runs]

        if len(runs)>0: return runs[-1] + 1
        else: return 1


    def plot_trajs(self, gen_trajs,epoch):
        plt.figure(figsize=(13, 2*len(gen_trajs)), dpi=60)
        for i, traj in enumerate(gen_trajs):
            plt.subplot(len(gen_trajs), 1, i+1)
            plt.plot(traj)
        plt.tight_layout()
        plt.savefig(self.dir_path+f'{epoch}_gen_traj.png', fmt='png', dpi=60)
        plt.close()


    def train(self, gen_iters, db_train, db_test):

        # salvo info log #
        with open(self.dir_path+'logs.txt','a+') as f:
            f.write((f"TRAINING\nbatch={self.batch_size}\nncritic={self.n_critic}\n"
                    f"gen_iterations={gen_iters}\ngen_lr={self.gen_lr} gen_b1={self.gen_b1}"
                    f" gen_b2={self.gen_b2}\ncritic_lr={self.critic_lr}"
                    f" critic_b1={self.critic_b1} critic_b2={self.critic_b2}\n"))
            f.write(self.text)
        log_fl = open(self.dir_path+'logs.txt','a+')
        fl = open(self.dir_path+'training.dat','a+')
        fl.write((f"# gen_iteration, critic_iteration, d_loss_tot, d_loss_true,"
                 f" d_loss_fake, d_loss_gp, d_loss_tot_test, d_loss_true_test,"
                 f" d_loss_fake_test, d_loss_gp_test, g_loss\n"))
        fl.close()

        # ############## #

        # static_noise = np.random.normal(0, 1, size=(1, self.noise_dim))
        d_loss_test = [0., 0., 0., 0.]
        g_loss = 0.
        M = db_train.max()
        m = db_train.min()
        semidisp = (M-m)/2.
        media = (M+m)/2.

        valid = -np.ones((self.batch_size, 1))
        fake =  np.ones((self.batch_size, 1))
        dummy = np.zeros((self.batch_size, 1)) # Dummy gt for gradient penalty

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
                noise = np.random.standard_t(4, size=(self.batch_size, self.noise_dim))
                # TEMP
                #noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim))
                #noise = np.random.uniform(-1., 1., (self.batch_size, self.noise_dim))
                # Train the critic
                # print("init disc train") 
                d_loss = self.critic_model.train_on_batch([imgs, noise],
                                                           [valid, fake, dummy])
                # print("disc trained")
                if (jj!=self.n_critic-1):
                        fl.write(("%7d %7d %11.4e %11.4e %11.4e %11.4e %11.4e"
                                  " %11.4e %11.4e %11.4e %11.4e\n")%(gen_iter,
                                  gen_iter*self.n_critic+jj,d_loss[0],d_loss[1],
                                  d_loss[2],d_loss[3],d_loss_test[0],
                                  d_loss_test[1],d_loss_test[2],d_loss_test[3],
                                  g_loss))

            # ---------------------
            #  Train Generator
            # ---------------------

            #idx = np.random.randint(0, db_test.shape[0], self.batch_size)
            #imgs = db_test[idx]
            #d_loss_test = self.critic_model.test_on_batch([imgs, noise],
            #                                              [valid, fake, dummy])
            ########### TEMP 29/08/20 14:53. UNUSED UNTIL RUN 9 ###############
            #noise = np.random.normal(0, 1, (self.batch_size, self.noise_dim)) #
            ###################################################################
            g_loss = self.gen_model.train_on_batch(noise, valid)
            # Plot the progress
            print((f"Gen_Iter: {gen_iter:6d} [D loss: {d_loss[0]:9.2e}] "
                  f"[d_loss_test: {d_loss_test[0]:9.2e}] "
                  f"[G loss: {g_loss:9.2e}]"))
            fl.write(("%7d %7d %11.4e %11.4e %11.4e %11.4e %11.4e "
                     "%11.4e %11.4e %11.4e %11.4e\n")%(
                    gen_iter,gen_iter*self.n_critic+jj,d_loss[0],d_loss[1],
                    d_loss[2],d_loss[3],d_loss_test[0],d_loss_test[1],
                    d_loss_test[2],d_loss_test[3],g_loss))
            fl.close()

            # If at save interval => save generated image samples
            if gen_iter % 100 == 0:
                self.plot_trajs(self.gen.predict(
                    np.random.normal(0,1, size=(3,self.noise_dim))),
                    gen_iter)
            if gen_iter % 250 == 0:
                self.critic.save(self.dir_path+f'{gen_iter}_critic.h5')
                self.gen.save(self.dir_path+f'{gen_iter}_gen.h5')
            if gen_iter % 2000 == 0:# and gen_iter > 0:    
                mini_db = self.gen.predict(
                    np.random.normal(0, 1, size=(50000, self.noise_dim)))
                mini_db = mini_db * semidisp + media
                np.save(self.dir_path+f'gen_trajs_{gen_iter}', mini_db)

                command_line = ("python ../compute_struct.py "+self.dir_path+
                                  f"gen_trajs_{gen_iter}.npy ../data/"+
                                  os.path.split(os.getcwd())[1]+
                                  f"/struct_function_{mini_db.shape[0]}_part_gen"+
                                  f"_{self.current_run}_{gen_iter}.npy")
                args = shlex.split(command_line)
                subprocess.Popen(args, stdout=log_fl, stderr=log_fl)
                command_line = ("python ../compute_pdf.py "+self.dir_path+
                                  f"gen_trajs_{gen_iter}.npy ../data/"+
                                  os.path.split(os.getcwd())[1]+
                                  f"/pdf0_{mini_db.shape[0]}_part_gen_"+
                                  f"{self.current_run}_{gen_iter}.npy")
                args = shlex.split(command_line)
                subprocess.Popen(args, stdout=log_fl, stderr=log_fl)
                command_line = ("python ../compute_pdf.py "+self.dir_path+
                                  f"gen_trajs_{gen_iter}.npy ../data/"+
                                  os.path.split(os.getcwd())[1]+
                                  f"/pdf1_{mini_db.shape[0]}_part_gen_"+
                                  f"{self.current_run}_{gen_iter}.npy -derivative")
                args = shlex.split(command_line)
                subprocess.Popen(args, stdout=log_fl, stderr=log_fl)
                del mini_db

        self.critic.save(self.dir_path+f'{gen_iter+1}_critic.h5')
        self.gen.save(self.dir_path+f'{gen_iter+1}_gen.h5')
        log_fl.close()


if __name__ == '__main__' :

    # ARGUMENTS AND OPTIONS PARSING

    import sys

    noise_dim = 100 # FIXED

    option_gen_iters = False
    gen_iters = 20000
    option_ncritic = False
    ncritic = 5
    option_load = False
    load = [0, 0]
    option_batch_size = False
    batch_size = 500
    option_gen_lr = False
    gen_lr = 0.00005
    option_critic_lr = False
    critic_lr = 0.0001

    i = 1
    while i < len(sys.argv):
        if sys.argv[i] == "-h":
            print((f"usage: wgan.py [--gen_iters <{gen_iters}>]"
                   f" [--ncritic <{ncritic}>] [--load <null> <null>]"
                   f" [--batch_size <{batch_size}>] [--gen_lr <{gen_lr}>]"
                   f" [--critic_lr <{critic_lr}>]"))
            exit()
        elif sys.argv[i] == "--gen_iters":
            option_gen_iters = True
            sys.argv.pop(i)
            gen_iters = int(sys.argv.pop(i))
        elif sys.argv[i] == "--ncritic":
            option_ncritic = True
            sys.argv.pop(i)
            ncritic = int(sys.argv.pop(i))
        elif sys.argv[i] == "--load":
            option_load = True
            sys.argv.pop(i)
            load[0] = int(sys.argv.pop(i))
            load[1] = int(sys.argv.pop(i))
        elif sys.argv[i] == "--batch_size":
            option_batch_size = True
            sys.argv.pop(i)
            batch_size = int(sys.argv.pop(i))
        elif sys.argv[i] == "--gen_lr":
            option_gen_lr = True
            sys.argv.pop(i)
            gen_lr = float(sys.argv.pop(i))
        elif sys.argv[i] == "--critic_lr":
            option_critic_lr = True
            sys.argv.pop(i)
            critic_lr = float(sys.argv.pop(i))
        else:
            print("arg not recognized, exiting ...")
            exit()
            i += 1


    # LOADING OR CREATING MODELS

    if option_load:

        run = load[0]
        number = load[1]
        path_gen = f'runs/{run}/{number}_gen.h5'
        path_critic = f'runs/{run}/{number}_critic.h5'
        gen = load_model(path_gen)
        critic = load_model(path_critic)
        critic.trainable = True
        # scrivo stringa info log gen #
        text = f"\nModels loaded. Continuing run {run} from number {number}\n"

    else:

        from gen import *
        from critic import *

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


    # WGANGP INIT

    wgan = WGANGP(gen, critic, noise_dim, ncritic, batch_size,
                  gen_lr, critic_lr, text)
    print((f'\nTrain for {gen_iters} generator iterations.\n'
           f"Parameters:\n    noise_dim: {noise_dim}\n    "
           f"ncritic: {ncritic}\n    batch_size: {batch_size}\n    "
           f"gen_lr: {gen_lr}\n    critic_lr: {critic_lr}\n"))
    print(text)


    # DB IMPORT

    print("\nImporting Databases ... ")
    db_train, db_test = load_data(1.0)


    # TRAINING

    print("\nTraining ... \n",end="")
    wgan.train(gen_iters, db_train, db_test)
