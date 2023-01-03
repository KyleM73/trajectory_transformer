import os
import sys
import argparse
import datetime
import numpy as np
import trajectory_transformer

parser = argparse.ArgumentParser(prog="trajectory_transformer",description="generate training data")
parser.add_argument("-d","--double_pendulum",help="simulates double pendulum",action="store_true")
parser.add_argument("n",help="number of trajectories to generate",type=int)
args = parser.parse_args()

if args.double_pendulum:
    from trajectory_transformer.envs import DoublePendulum
    env = DoublePendulum(record=True)
    data_dir = trajectory_transformer.data.path+"/double_pendulum"
    cfg = {
        "m1" : 1,
        "l1" : 1,
        "m2" : 1,
        "l2" : 1,
        "g" : 9.8,
        "u_max" : 10,
        "dt": 0.001,
        "horizon" : 10
        }
## other envs go here
else:
    print("Error: no env provided")
    sys.exit(1)

data_dir += "/{}".format(datetime.datetime.now().strftime('%m%d_%H%M'))
os.mkdir(data_dir)

for i in range(args.n):
    fname = data_dir+"/trial{}.csv".format(i)
    #fname_gz = data_dir+"/trial{}.gz".format(i)

    env.reset()

    for i in range(env.T-1):
        action = np.zeros((env.action_space.shape))
        env.step(action)

    data = env.get_hist()[0]

    np.savetxt(fname,data,delimiter=",")
    #np.savetxt(fname_gz,data,delimiter=",")








