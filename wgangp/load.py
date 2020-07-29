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
print("\n Loading generator ... ", path_gen)
save_path = f'runs/{run}/gen_trajs_{number}'
print("\n Save path : ", path_gen)
#path_critic = f'runs/{run}/{number}_critic.h5'
#print(path_critic)

gen = load_model(path_gen)

noise_dim = 100
noise = np.random.normal(0, 1, (500000, noise_dim))

trajs = gen.predict(noise)
np.save(save_path, trajs)


print("Done !") 
