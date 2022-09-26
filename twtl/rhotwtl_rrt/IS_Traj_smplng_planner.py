'''
Created on Aug 20, 2016

@author: cristi
'''

# from cmath import inf
import logging
import itertools as it
from re import T
import time
from collections import deque
import time
import timeit
import math

import numpy as np
from numpy.random import uniform, exponential
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm
import plotly.graph_objects as go
import normflows
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvnrnd 
from cem_mvnrnd import CEM
torch.manual_seed(10)




bernoulli = lambda p=0.5: 0 if uniform() <= p else 1
# import networkx as nx

# from lomap import Ts

from stl import STL
from hscc2017_specs import get_specification_ast
from mission import Mission
from systems import load_system

DBL_MAX = 100000.0
global N
global current_k
global epsilon




class Planner(object):
    '''Class for planner.'''

    def __init__(self, system, specification, params,mission,CEM_mvnrnd):
        '''Constructor'''
        self.system = system
        self.specification = specification

        # self.ts.init = self.system.initial_state

        self.steps = params['planning_steps']
        self.gamma = params['gamma']
        self.save_plt_flg = params['save_plt_flg']
        self.mission = mission

        # if params['choice_function'] == 'random_simple':
        #     self.choose = choose_random
        # else:
        #     raise ValueError('Unknown choice function "%s"',
        #                      params['choice_function'])

        self.T_H = 1
        self.min_step = 0.01
        assert self.min_step < self.T_H
        self.cem_mvnrnd = CEM_mvnrnd
        # Importance sampling attributes: 
        self.ext_set = []
        self.elite_set = []
        #Previous and current addaptation parama
        self.adapIter = 1
        self.kde_preSamples = []
        self.kde_currSamples =[]
        self.curr_Ldist = 0
        self.prev_Ldist = 0
        #Reaching the optimal distribution params:
        #---kde
        self.kdeOpt_flag = False
        self.kde_eliteSamples = []
        self.KDE_fitSamples = None
        self.KDE_pre_gridProbs = None
        self.kde_enabled = True
        self.initEliteSamples = []
        self.len_frakX = 0
        self.pre_gridProbs = []
        self.qt_rhos = []
        self.best_rhos = []
        self.plot_kde_est = True

        # Plotting stuff: 
        self.plt_ext_flg = True

    ##############
    ############
    #********************
    # Sampling in the trajectroy space: 
    #********************  
    def sample_us(self,Td, n_s = 6): 
        ''' 
        Given the robot dynamics, this function generates a sequence of "n_s" control inputs, each of which is applied for "T". 
        "n_s" is a tuning param, such that, Td = n_s * T  
        Td: time duration of the generated control inputs
        n_s: 
        '''
        # TODO [code_opt] - Is it more efficient to return an explicit control sequence with respect to the simulation time step (dt) [could be used in evolving the dynamics with vectorized computation]? 
        #                 - Is it more efficient to return control actions primitives  
        u_bounds = self.control_bounds
        lb = u_bounds[0]/2.5
        ub = u_bounds[1]/2.5
        # lb = 0 
        # ub = 1.5 
        T = Td*1.0/n_s # the time duration for each control input of the control sequence  
        dt = self.dt
        n_i = math.ceil(T/dt) # Number of applying the same control input 
        map_bnd = lambda x: (ub-lb)*x + (lb)
        vfunc = np.vectorize(map_bnd)
        u_seq = np.array([])
        for i in range(n_s): 
            #Uniform sampling, TODO [traj_smplng] we could play as well with the sampler of the control sequence 
            u_rand = vfunc(np.repeat(np.random.sample(1),n_i))
            u_seq = np.concatenate((u_seq,u_rand),axis=None)
        return list(u_seq)
    #********************
    #********************
    ##############
    ##############

    def solve(self):
        '''
        TODO: - Inspect every newlly added edge to the tree, 
        '''
        # self.initialize()

        runtimes = []
        rhos = [] 
        extensions_set = []
       
       # Go to CEM and return the trajectroy that maximizes the robustness: 
                # - Function to compute the STL robustness --- DONE 
                # - Function to perform CEM -- done for KDE with state space sampling, TODO do it using GMM for the motion primitives 
                # - Sampler from GMM, TODO
                #  
        # Generating (random) control primitives from GMM:
        n_trajs = 70 
        n_cem_iters = 50 # @ each iteration I'll have 50 trajectories 
        n_u_primv = 5 
        q = 0.9
        v = .8 #smoothing parameter to prevent degenerate estimation
        cem_iter_smpls = torch.zeros(n_u_primv,n_trajs) # Preallocate tensor for the samples 
        # Inetial samples: 
        # for i in range(n_trajs):
        #     u_primv = self.system.init_u_primv(n_primv = n_u_primv) # 
        #     cem_iter_smpls[:,i] = u_primv
        #     a = 1
        for i in range(n_cem_iters):
            # Generate motion primitives (while you are generating the trajectories, you might as well draw them in real-time): 
            cem_iter_smpls = self.cem_mvnrnd.sample(n_smpls = n_trajs)
            #  u_primvs_set = None, n_trajs = None, STL_specs = None, cost_flg = False, Th = None
            trajs, rhos = self.system.trajs(u_primvs_set = cem_iter_smpls, n_trajs = n_trajs, STL_specs = self.specification, Th = self.specification.bound)            
            #Assign the elite set of control actions: 
            rho_rhoth_q = torch.quantile(input=rhos,q=q) 
            inds = [i for i, rho in enumerate(rhos) if rho>=rho_rhoth_q]
            elite_trajs = trajs[:,:,inds]
            elite_ctrls = cem_iter_smpls[:,inds]
            # Adapt mu and cov of the GMM: 
            mu = (1-v) *  self.cem_mvnrnd.mu + v* torch.mean(elite_ctrls,1)
            cov = (1-v) * self.cem_mvnrnd.cov + v* torch.cov(elite_ctrls)
            self.cem_mvnrnd.update_gmm_params(mu = mu, cov = cov)
            self.qt_rhos.append(rho_rhoth_q) # Just for displaying the results
            self.best_rhos.append(max(rhos)) # Just for displaying the results


            
            # TODO [w@]: - Compute the stl robustness (done)
            #            - Extract the elite samples (done)
            #            - Compute the mu and cov using EM 

            # Adapt mu and 


            pass


        a = 1
        # for i in range(1000): # The CEM loop: 
        #     u_seq = self.system.sample_us(Td=Td,n_s=n_s)
        #     u, traj,xf = self.system.steer_undrU(x0 = x0,ti = ti,u_seq=u_seq)
        #     traj = [s_traj,t_traj]
        #     rho = self.specification.robustness(traj)
        #     rhos.append(rho)
        #     traj_rho_pr = [traj,rho]
        #     extensions_set.append(traj_rho_pr)
       
        
        if len(self.ts.nodes) % 100 == 0: #plot the tree every 10 iterations: 
            debug_show_ts(self, self.mission.system['type'])
            a = 1

        with open('runtimes.txt', 'a') as fout:
            print>>fout, runtimes
            print>>fout

        # save solution
        if self.exists_solution():
            self.ts.best_path = self.get_path_from_root(self.lowerBoundVertex)
            self.ts.solution_controls = [self.ts.nodes[x][phi]['control']
                                                for x, phi in self.ts.best_path]
            return True
        return False

    
    def exists_solution(self):
        '''Checks if there exists a solution.'''
        return self.lowerBoundVertex is not None

    def get_path_from_root(self, v):
        '''Returns the path from root to v.'''
        print('vertex:', v)
        traj = deque([])
        while v:
            traj.appendleft(v)
            x, phi = v
            v = self.ts.nodes[x][phi]['parent']
        return traj


def debug_show_ts(planner, system_type):
    import os.path
    # import matplotlib.pyplot as plt

    figure = plt.figure()
    # TODO [PY2]>>>
    # figure.add_subplot('111')
    # TODO [PY2]<<<
    # TODO [PY3]>>>
    figure.add_subplot(111)
    # TODO [PY3]<<<
#     axes = figure.axes[0]
#     axes.axis('equal') # sets aspect ration to 1

    plt.xlim(planner.system.bound[0])
    plt.ylim(planner.system.bound[1])

    # draw regions
    if system_type == 'double_integrator':
        x, y = zip(*planner.ts.nodes)
        # # AV specs: 
        # plt.fill([0.1, 0.4, 0.4, 0.1, 0.1], [.1, .1, 0.2, 0.2, .1],
        #    color=(1, 0.5, 0, 0.2)) # Start Acce
        # plt.fill([3.5, 4, 4, 3.5,3.5], [.1, .1, 0.3, 0.3, .1],
        #    color=(1, 0.5, 0, 0.2)) # Done with passing
        # plt.fill([1.8, 2.5,2.5, 1.8,1.8], [-.1, -.1, 0.1, 0.1, -.1],
        #    color=(1., .1, .1,.1)) # Obstacle

        # #The Always Regions: 
        # plt.fill([.4, 1, 1, .4, .4], [.15, .15, .5, 0.5, .15],
        #          color=(0, 1, 0, 0.2))
        # plt.fill([1.5, 3, 3, 1.5, 1.5], [.3, .3, .7, 0.7, .3],
        #          color=(0, 1, 0, 0.2))
        # plt.fill([3.5, 4, 4, 3.5,3.5], [.1, .1, .3, .3, .1],
        #          color=(0, 1, 0, 0.2))
         

        
        #$$$ Cristi's Specs: 
        plt.fill([3.5, 4, 4, 3.5, 3.5], [-0.2, -0.2, 0.2, 0.2, -0.2],
             color=(1, 0.5, 0, 0.2))
        plt.fill([0, 2.1, 2.1, 0, 0], [-0.5, -0.5, 0.5, 0.5, -0.5],
                 color=(0, 0, 1, 0.2))

        plt.fill([2, 3, 3, 2, 2], [0.5, 0.5, 1.0, 1.0, 0.5],
                 color=(0, 1, 0, 0.2))
    elif system_type == 'rear_wheel_car':
        axes = figure.axes[0]
        axes.axis('equal') # sets aspect ration to 1

        x, y, _, _, _= zip(*planner.ts.nodes)
        plt.fill([1, 1, 2, 2, 1], [2, 3, 3, 2, 2], color=(0.7, 0.7, 0.7, 0.2))
        plt.fill([3, 4, 4, 3, 3], [2, 2, 3, 3, 2], color=(1, 0.5, 0, 0.2))
#         plt.fill([2.5, 3.5, 3.5, 2.5, 2.5], [1, 1, 2, 2, 1],
#                  color=(0.7, 0.7, 0.7, 0.2))
#         plt.fill([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], color=(0, 0.5, 1, 0.2))
    else:
        raise NotImplementedError

    plt.xlabel(planner.specification.space.var_names[0])
    plt.ylabel(planner.specification.space.var_names[1])
    plt.plot(x, y, 'ko')
    xi, yi = planner.system.initial_state[:2]
    plt.plot([xi], [yi], 'bD')

    for x in planner.ts.nodes:
        # phi_d = planner.ts.nodes[x]
        
        for phi in planner.ts.nodes[x]:
            # Plotting the node trajectory and the DIS: 
            traj, _ = planner.ts.nodes[x][phi]['trajectory']
            if system_type == 'double_integrator':
                xx, yy = zip(*traj)
            elif system_type == 'rear_wheel_car':
                xx, yy,_, _, _ = zip(*traj)
            plt.plot(xx, yy, 'k')
            #Plot DIS of each node
            #------------------------------------------------
            # u,v = planner.ts.nodes[x][phi]['chi']
            # plt.quiver(x[0],x[1],u,v,width = 0.003,color = 'g')

            # Plot the exploratory trajectory -- if one exists: 
            #----------------------------------------------
            if planner.plt_ext_flg:
                ext_traj = planner.ts.nodes[x][phi]['ext_traj']
                ext_traj = ext_traj[1][0]
                if type(ext_traj) is list:
                    xx_ext, yy_ext = zip(*ext_traj)
                    plt.plot(xx_ext, yy_ext, 'g')
                a = 1
        


    if planner.exists_solution():
        v = planner.ts.best_path.pop()
        while v:
            x, phi = v
            plt.plot([x[0]], [x[1]], 'rd')
            v_pa = planner.ts.nodes[x][phi]['parent']
            if v_pa:
                x_pa, _ = v_pa
                plt.plot([x[0], x_pa[0]], [x[1], x_pa[1]], 'r')
            v = v_pa

    plt.tight_layout()
    script_dir = os.path.dirname(__file__)
    out_dir = os.path.join(script_dir, 'cases/')
    if planner.save_plt_flg:
        for ext in ('png', 'jpg', 'eps'):
            plt.savefig(os.path.join(out_dir, '{}_{}.{}'.format(system_type,
                                                            planner.steps, ext)))
    plt.show()


def main():
    import sys, os.path
    if len(sys.argv) > 1:
    # load mission file
        if sys.argv[1]=='di':
            filename = '../cases/case1_di.yaml'
        elif sys.argv[1]=='rwc':
            filename = '../cases/case2_rwc.yaml'
        else:
            raise NotImplementedError
    else:
#         filename = '../cases/case2_rwc.yaml'
        filename = '../cases/case1_di.yaml'

    # setting up logging
    loglevel = logging.DEBUG
    fs, dfs = '%(asctime)s %(levelname)s %(message)s', '%m/%d/%Y %I:%M:%S %p'
    logging.basicConfig(stream=sys.stdout, level=loglevel, format=fs,
                        datefmt=dfs)

    mission = Mission.from_file(filename)

    # set seed
    np.random.seed(mission.planning['seed'])

    # load system model
    system = load_system(mission.system)

    # load specification
    specification = get_specification_ast(mission.specification)
    logging.info('Specification time bound: %f', specification.bound)

    global N
    global current_k
    global epsilon

    N = mission.planning['planning_steps']
    epsilon = 0.1
    current_k = 0
    CEM_mvnrnd = CEM(dim = 5)
    planner = Planner(system, specification, mission.planning,mission,CEM_mvnrnd=CEM_mvnrnd)

    logging.info('Start solving:')

    found = planner.solve()
    for x in planner.ts.nodes:
        for phi in planner.ts.nodes[x]:
            print(planner.ts.nodes[x][phi]['rosi']),
            print(planner.ts.nodes[x][phi]['time'])

    if found:
        logging.info('Found solution!')
        logging.info('max robustness: %f', planner.lowerBoundCost)
        logging.info('RoSI: %s', str([planner.ts.nodes[x][phi]['rosi']
                                          for x, phi in planner.ts.best_path]))

    out_dir = '../cases'
    ts_filename = os.path.join(out_dir, mission.planning['solution_filename'])
    # with open(ts_filename, 'w') as fout:
    #     print>>fout, planner.ts.nodes

    debug_show_ts(planner, mission.system['type'])

    logging.info('Done!')

if __name__ == '__main__':
    main()
