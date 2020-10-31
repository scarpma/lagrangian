#!/usr/bin/env python
# coding: utf-8

import subprocess as sp
import os
from params import *
import sys

from datetime import datetime
now = datetime.now()
dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
print("ESECUZIONE: ", dt_string, "\n")

run = 51
number = 1750
batchs_per_iter = 10 # va moltiplicato per batch_size (di solito 50k) 
iters = 100
traj_path = (f"/storage/scarpolini/databases/"+DB_NAME+"/"+
         WGAN_TYPE+f"/runs/{run}/gen_trajs_{number}.npy")

for i in range(iters):
    if os.path.exists(traj_path):
        os.remove(traj_path)
    print(f"Iteration n: {i}")
    print("\tGenerating trajs ... ",end="")
    cmd_traj = ["python", "traj_gen.py", str(run), str(number),
                str(batchs_per_iter)]
    p_traj = sp.Popen(cmd_traj, stdout=sp.PIPE)
    (out_traj, err_traj) = p_traj.communicate()
    # if err_traj != None:
    #     print("some error occured, exiting...")
    #     exit()
    print("Output: ", out_traj, "\n", err_traj)
    print("done.")

    cmd_pdf = ["python", "graphics/plot_pdfs.py", str(run), str(number),
               "--only_hist"]
    p_pdf = sp.Popen(cmd_pdf, stdout=sp.PIPE)
    cmd_sf = ["python", "graphics/plot_sf.py", str(run), str(number)]
    p_sf = sp.Popen(cmd_sf, stdout=sp.PIPE)

    (out_sf, err_sf) = p_sf.communicate()
    (out_pdf, err_pdf) = p_pdf.communicate()

    pdf_status = p_pdf.wait()
    print("pdfs done.")
    sf_status = p_sf.wait()
    print("sfs done.")

    if os.path.exists(traj_path):
        os.remove(traj_path)
    print("trajs removed.")


print("Iterations done. Doing last step")
cmd_pdf = ["python", "graphics/plot_pdfs.py", str(run), str(number),
           "--read_gen"]
sp.Popen(cmd_pdf, stdout=sp.PIPE)
(out, err) = sp.communicate()
print(out, err)
cmd_sf = ["python", "graphics/plot_sf.py", str(run), str(number),
          "--read_gen"]
sp.Popen(cmd_sf, stdout=sp.PIPE)
(out, err) = sp.communicate()
print(out, err)
