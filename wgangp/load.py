#!/usr/bin/env python
# coding: utf-8

from db_utils import *
from wgan import *

parser = argparse.ArgumentParser()
parser.add_argument('run', type=int)
parser.add_argument('number', type=int)
args = parser.parse_args()

run = args.run
number = args.number

path_gen = f'runs/{run}/{number}_gen.h5'
path_critic = f'runs/{run}/{number}_critic.h5'

print(path_gen)
print(path_critic)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

tensorflow.keras.losses.wasserstein_loss = wasserstein_loss
gen = load_model(path_gen, custom_objects={'wasserstein_loss': wasserstein_loss})
critic = load_model(path_critic, custom_objects={'wasserstein_loss': wasserstein_loss})
noise_dim = 100
ncritic = 5
wgan = WGANGP(gen, critic, noise_dim, ncritic, 250, '0')
db_train , _ = load_data()
#wgan.train(3000, db_train, db_test)

idx = np.random.randint(0,db_train.shape[0], wgan.batch_size)
noise = np.random.normal(0, 1, (wgan.batch_size, wgan.noise_dim))
pred_real = wgan.critic.predict(db_train[idx])
pred_fake = wgan.critic.predict(wgan.gen.predict(noise))
plt.plot(pred_real,label='real')
plt.plot(pred_fake,label='fake')
plt.savefig(f'runs/{run}/pred.png', fmt='png', dpi=100)
plt.close()

trajs = wgan.gen.predict(np.random.normal(0, 1, size=(50000, noise_dim)))
np.save(f'runs/{run}/gen_trajs', trajs)
