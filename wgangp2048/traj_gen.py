from db_utils import *

run = 56
number = 4500

path = f'/scratch/scarpolini/lagrangian/wgangp/runs/{run}/{number}_gen.h5'
print('Loading Model ...')
gen = load_model(path)


N = 1
bs = 50000
trajs = np.zeros(shape=(N*bs,2048,1)) #VAR
print('Generating Trajectories ...')
for ii in range(N):
    print(ii)
    noise = np.random.normal(0, 1, size=(bs, 100)) #VAR
    trajs[ii*bs:(ii+1)*bs,:,0:1] = gen.predict(noise, verbose=1, batch_size=bs)

try: os.mkdir((f'/storage/scarpolini/databases/lagrangian/"
               f"wgangp2048/runs/{run}')) #VAR

except: print('Directory already exists')
print('Saving ...')
#VAR
np.save((f'/storage/scarpolini/databases/lagrangian/"
         f"wgangp2048/runs/{run}/gen_trajs_{number}', trajs))

