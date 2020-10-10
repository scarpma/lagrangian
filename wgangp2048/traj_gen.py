import sys

# COMMAND LINE PARSING

renorm = True

if sys.argv[1] == '-h' :
    print("usage: traj_gen.py <run> <number> <n_batch> [--no_renorm]")
    exit()
if sys.argv[1] == '--no_renorm' :
    renorm = False
    sys.argv.pop(1)
elif len(sys.argv) < 2:
    print("usage: traj_gen.py <run> <number> <n_batch>")
    exit()

run = int(sys.argv.pop(1))
number = int(sys.argv.pop(1))
N = int(sys.argv.pop(1))

while len(sys.argv) > 1 :
    print('Unrecognized args. Exiting.')
    exit()

from db_utils import *

path = f'runs/{run}/{number}_gen.h5'
print('Loading Model ...')
gen = load_model(path)

bs = 50000
trajs = np.zeros(shape=(N*bs,SIG_LEN,CHANNELS))
print('Generating Trajectories ...')
for ii in range(N):
    print(ii)
    # noise = np.random.normal(0, 1, size=(bs, NOISE_DIM)) #VAR
    noise = np.random.standard_t(4, size=(bs, NOISE_DIM)) #VAR
    # noise = np.random.standard_t(3.5, size=(bs, NOISE_DIM)) #VAR
    trajs[ii*bs:(ii+1)*bs,:,0:1] = gen.predict(noise, verbose=1, batch_size=bs)

if renorm:
    semidisp = (DB_MAX - DB_MIN)/2.
    media = (DB_MAX + DB_MIN)/2.
    trajs = trajs * semidisp + media

try: os.mkdir((f"/storage/scarpolini/databases/"+DB_NAME+"/"+
               WGAN_TYPE+f"/runs/{run}")) #VAR

except: print('Directory already exists')
print('Saving ...')
#VAR
np.save((f"/storage/scarpolini/databases/"+DB_NAME+"/"+
         WGAN_TYPE+f"/runs/{run}/gen_trajs_{number}"), trajs)

