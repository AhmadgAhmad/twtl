'''
Created on 2022

@author: Ahmad Ahmad

This code is a modified version of STL-RRT* [Cristi 17] in which 
we implement TWLTL-RRT*. 
TWTL-RRT*: Grows a tree to find a path that maximizes the robust
satisfaction of a TWTL formula. 


TODO: 
- develop the runtime robustness monitoring algorithm of TWTL
- define RoIS of TWTL 
- 

'''

# from cmath import inf
import logging
import itertools as it
from re import T
import time
from collections import deque
import time
import timeit

import numpy as np
from numpy.random import uniform, exponential
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm
import plotly.graph_objects as go
import twtl.twtl  as twtl
from antlr4 import InputStream, CommonTokenStream
from twtlLexer import twtlLexer
from twtlParser import twtlParser



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


def blend(x1, x2):
    '''Simple blending function.'''
    x1n = np.asarray(x1)
    x2n = np.asarray(x2)
    if np.isclose(np.dot(x1n, x2n), 0, rtol=0):
        return tuple(x1n + x2n)
    return x1

def choose_random(x1, x2, I1, I2, rho_max):
    '''Simple stochastic sampling function.'''
    if I1[0] <= I2[0] and I1[1] <= I2[1]:
        return x1, I1, x2, I2
    elif I1[0] >= I2[0] and I1[1] >= I2[1]:
        return x2, I2, x1, I1
    else:
        p = 0.5 + (I1[0] + I1[1] - I2[0] - I2[1]) / (8.0 * rho_max)
        if bernoulli(p) == 0:
            return x1, I1, x2, I2
        else:
            return x2, I2, x1, I1

class Tree(object):
    '''TODO:'''
    def __init__(self):
        self.nodes = dict()
        self.init = None

    def add_node(self, state, attr_dict):
        self.nodes[state] = attr_dict


class Planner(object):
    '''Class for planner.'''

    def __init__(self, system, specification, params,mission):
        '''Constructor'''
        self.system = system
        self.specification = specification

        # initialize RRT tree
        self.ts = Tree()
        self.ts.init = self.system.initial_state

        self.steps = params['planning_steps']
        self.gamma = params['gamma']
        self.save_plt_flg = params['save_plt_flg']
        self.mission = mission

        if params['choice_function'] == 'random_simple':
            self.choose = choose_random
        else:
            raise ValueError('Unknown choice function "%s"',
                             params['choice_function'])

        self.T_H = 1
        self.min_step = 0.01
        assert self.min_step < self.T_H

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
    ##############
    def KLdiv(self,grid_probs):
        """
        Compute the KL divergence
        :param grid_probs: The probabilities of the point in the grid of the current sampling distribution
        :return: the KL divergence
        """
        if self.kde_enabled:#self.params.kde_enabled:
            pre_grid_probs = self.KDE_pre_gridProbs
        else:
            pre_grid_probs = self.pre_gridProbs
        return -sum([pre_grid_probs[i]*np.log2(grid_probs[i]/pre_grid_probs[i]) for i in range(len(pre_grid_probs))])

    def UpSampling_andSample(self, extensions_set,length = 1,width = 4, rho = 0.1, rce = .5 ,md=8, k=4, n=2):
            u_rand = uniform(0, 1)
            if (len(extensions_set) > 30):# and len(extensions_set)>20: #Exploration w.p. 0.5 and exploitation (sampling adaptation) w.p. 0.5
                N_qSamples = 200 
                h = 10.0/md

                #Optimality flags, and distribution fitting options
                if (self.kde_enabled and not self.kdeOpt_flag):
                    t_s, x_s = self.CE_Sample(extensions_set,h,N_qSamples)
                elif self.kde_enabled and self.kdeOpt_flag: #The optimal kernel density estimate has been reached, focus the samples just to be for the "optimal" distribution
                    t_s, x_s = self.ufrm_sample()
                if t_s is None and x_s is None: #CE_Sample() will return None if not enough samples of trajectoies that reach the goal are avialbe
                    t_s, x_s = self.ufrm_sample()
            else:
                t_s, x_s = self.ufrm_sample()
            return t_s, x_s

    def CE_Sample(self,extensions_set,h,N_qSamples):   
        if len(extensions_set)>=(self.adapIter*50): #The acceptable number of trajectories to adapat upon
            frakX = []
            #Find the elite trajectoies then discretize them and use their samples as the elite samples:
            extensions_set_rhoList = list(np.asarray(extensions_set)[:,1]) 
            d_factor = 1
            q = 0.25/d_factor # TODO [elite] the %70 samples
            rho_rhoth_q = np.quantile(list(np.multiply(extensions_set_rhoList,-1.0)), q=q)
            self.qt_rhos.append(-rho_rhoth_q)
            elite_extensions_set = [traj for traj in extensions_set if -traj[1] <= rho_rhoth_q] # TODO [elite] iterate upon pairs, or, as opposed to define pairs, I might as well define the  
            elite_extensions_set_rhos = [traj[1] for traj in extensions_set if -traj[1] <= rho_rhoth_q]
            self.best_rhos.append(max(elite_extensions_set_rhos))
            if len(elite_extensions_set) == 0:
                elite_extensions_set = extensions_set

            #XXXXXXX
            for elite_traj in elite_extensions_set:
                elite_traj_rho = elite_traj[1]
                #Concatnating the trajectory:
                traj = np.asarray(elite_traj[0][0])
                t_traj = np.asarray(elite_traj[0][1])
                # Backtrack the path from vg to v0; extract the sample at certain increments of the time:
                tStep_init1 = int(h/self.system.dt)
                tStep_init = 2
                tStep = 10
                tStep_temp = tStep_init1+3
                while tStep < len(traj[:,1]):
                    pi_qt_tpl = np.append(traj[tStep,:],t_traj[tStep])# The time of the elite system state (TODO [traj_smplng] When sampling form the space of trajectories, the temporal coherence is within implied by the trajectory itself.#TODO [q_ahmad] how the space of trajectories would look like. )
                    elite_cddtSample = [pi_qt_tpl,elite_traj_rho] #This tuple contains the actual sample pi_q_tStep and the CostToCome to the goal of the corresponding trajectory
                    frakX.append(elite_cddtSample)  # TODO [smplng] Don't ignore the previous iterations elite distributions. See iCEM for their approach in carrying some previous elite samples. Employ the idea of ergodic exploration as opposed to ignore all the previous samples 
                    tStep = tStep + tStep_temp
            if self.adapIter == 1:
                frakX.extend(self.initEliteSamples)
            # XXXXXXX
            self.len_frakX = len(frakX)
            if len(frakX) == 0:
                ok = 1
            t_s,x_s = self.CE_KDE_Sampling(frakX) # TODO [smplng] Here, do the normalizing flow sampling 
        else:
                t_s = None
                x_s = None
        return t_s,x_s

    #Density estimate, kernel density or GMM:
    def CE_KDE_Sampling(self,frakX):
        """
        Fit the elite samples to Kernel density estimate (KDE) or a GMM to generate from; and generate an (x,y) sample from the estimated
        distribution. Checks if the CE between the previous density estimate and the current one below some threshold. In the case
        of KDE the expectation similarity measure could be used instead on the CE.

        NOTE to Ahmad:
        You're using the CE with the KDE because you have the logistic probes of the samples and you use them; however, for
        Kernel based distributions the expectation similarity could be used as well. One might reformulate the CE framework
        in terms of nonparametric distributions.

        :param frakX: The elite set of samples with the corresponding trajectory cost.
        :return:
        """
        frakXarr = np.array(frakX)
        N_samples = len(frakX)
        costs_arr = frakXarr[:,1]
        elite_samples_arr = np.asarray(list(frakXarr[:,0]))
        elite_costs = costs_arr

        #random point from the estimated distribution:
        if self.kde_enabled:#self.params.kde_enabled:
            kde = KernelDensity(kernel='gaussian', bandwidth=.4) # TODO [smplng] choose the bandwidth differently  
            kde.fit(elite_samples_arr)
            self.adapIter += 1
            xySample = kde.sample()

        if self.kde_enabled:#self.params.kde_enabled:
            x_gridv = np.linspace(-1, 5, 50)
            y_gridv = np.linspace(-1, 4, 50)
            t_gridv = np.linspace(0,12,50) # TODO [smplng] Choose the time horizon of the STL specs 
            Xxgrid, Xygrid, Xtgrid = np.meshgrid(x_gridv, y_gridv, t_gridv)
            XYTgrid_mtx = np.array([Xxgrid.ravel(), Xygrid.ravel(),Xtgrid.ravel()]).T
            #Get the probabilities
            grid_probs = np.exp(kde.score_samples(XYTgrid_mtx))
            elite_smpls_probs = np.exp(kde.score_samples(elite_samples_arr))
            # Find the KL divergence the current samples and the previous ones:
            if self.adapIter > 2:
                KL_div = self.KLdiv(grid_probs)
                if KL_div < .1:
                    # self.kdeOpt_flag = True
                    pass
                    
                self.KDE_fitSamples = kde #This kde object will be used to sample form whn the optimal sampling distribution has been reached

            self.KDE_pre_gridProbs = grid_probs

            #Plot the distribution
            if self.plot_kde_est:
                # self.initialize_graphPlot()
                # CS = plt.contour(Xxgrid, Xygrid,Xtgrid, grid_probs.reshape(Xxgrid.shape))  # , norm=LogNorm(vmin=4.18, vmax=267.1))
                # plt.colorbar(CS, shrink=0.8, extend='both')
                
                # Visualize the isosurfaces of the 3d multivariate distribution: 
                #---------------------------------------------------------------
                fig = go.Figure(data=go.Isosurface(
                    x = Xxgrid.flatten(),
                    y = Xygrid.flatten(),
                    z = Xtgrid.flatten(),
                    value = grid_probs,
                    opacity=0.6,
                    surface_count=7, # number of isosurfaces, 2 by default: only min and max
                    colorbar_nticks=7, # colorbar ticks correspond to isosurface values
                    caps=dict(x_show=False, y_show=False)
                    ))
                
                fig.show()

                # Visualize the elite samples (scatter points)
                #------------------------------------------------------
                fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
                ax.scatter(elite_samples_arr[:, 0], elite_samples_arr[:, 1],elite_samples_arr[:, 2],c=elite_smpls_probs)
                ax.set_xlabel('x_pos')
                ax.set_ylabel('y_pos')
                ax.set_zlabel('time')
                plt.show()
                
                # Visualize the contours of the multivariate distribution at a time instance: 
                # ---------------------------------------------------------------------------- 
                CS = plt.contour(Xxgrid[:,:,49], Xygrid[:,:,49], grid_probs.reshape(Xxgrid.shape)[:,:,49])  # , norm=LogNorm(vmin=4.18, vmax=267.1))
                plt.colorbar(CS, shrink=0.8, extend='both')
                
                a = 1

        return xySample[0][2],(xySample[0][0],xySample[0][1])
    ##############
    ##############

    def solve(self):
        '''
        TODO: - Inspect every newlly added edge to the tree, 
        '''
        self.initialize()

        runtimes = []
        rhos = [] 
        extensions_set = []
        # =========== The planning loop (Line 3 - 18 Algorithm 1) ================
        for k in range(10000):#range(self.steps):
            current_k = k
            if not k%100:
                logging.info('step: %d %d', k, len(self.ts.nodes))
            t0 = time.time()

#             print '----------------------------------'

            t_rand, x_rand = self.ufrm_sample()
            # _,_ =self.UpSampling_andSample(extensions_set=extensions_set)
            
            
#             print 'sample:', t_rand, x_rand

            N_near = self.near(x_rand, t_rand)
            if not N_near:
                x_pa, t_pa, t_rand = self.nearest(x_rand, t_rand)
                if x_pa is None:
                    continue
                assert t_pa + self.min_step <= t_rand
                _, _, x_rand = self.system.steer(x_pa, x_rand, t_rand-t_pa, t_pa)
#                 assert self.system.dist(x_pa, x_rand) <= self.gamma
                if self.system.dist(x_pa, x_rand) > self.gamma:
                    print('nearest-parent:', t_pa, x_pa)
                    print('nearest-steer:', t_rand, x_rand)
                    print('dist:', self.system.dist(x_pa, x_rand))
                    assert False, (self.system.dist(x_pa, x_rand), self.gamma,
                                   self.system.dist(x_pa, x_rand) <= self.gamma)
                N_near = self.near(x_rand, t_rand)
            if not N_near:
                print(x_rand)
                print(t_rand)

                X_near = [x for x in self.ts.nodes
                                if self.system.dist(x_rand, x) <= self.gamma]
                for xx in X_near:
                    print(xx)
                    for phi, d in self.ts.nodes[xx].iteritmes():
                        print(phi, d['time'])
                    print

            assert N_near

#             print t_rand, x_rand, N_near

            # drive system towards random state and increasing robustness
            # TODO [ahmad] Here the things are 
            v_new, connected = self.connect(x_rand, t_rand, N_near)

#             print v_new
#             print connected
            # connected = False
            if connected:
                self.rewire(v_new, t_rand)
                # -------------- Extend a vertex T_H ---------------------------
                x0 = v_new[0]
                ti = self.ts.nodes[v_new[0]][v_new[1]]['time']
                Td = 10 - (ti)
                n_s = 20
                for isple in range(1):
                    u_seq = self.system.sample_us(Td=Td,n_s=n_s)
                    u, traj,xf = self.system.steer_undrU(x0 = x0,ti = ti,u_seq=u_seq)
                    self.ts.nodes[v_new[0]][v_new[1]]['ext_traj'] = [u, traj,xf]
                    v_traj =  self.ts.nodes[v_new[0]][v_new[1]]['trajectory']
                    t_traj = v_traj[1] + traj[1] # Appending the time trajectory
                    s_traj = v_traj[0] + traj[0] # Appending the state trajectory 
                    traj = [s_traj,t_traj]
                    rho = self.specification.robustness(traj)
                    rhos.append(rho)
                    traj_rho_pr = [traj,rho]
                    extensions_set.append(traj_rho_pr)
                self.UpSampling_andSample(extensions_set=extensions_set) # TODO [smplng] remove from here 
                # TODO [wrat] Compute STL robustness of each extension:
                 

                # --------------------------------------------------------------

            # $$$$$$$ TODO [ahmad] Inspect the tree 
            if len(self.ts.nodes) % 100 == 0: #plot the tree every 10 iterations: 
                debug_show_ts(self, self.mission.system['type'])
                a = 1
            nodes_inspe = self.ts.nodes
            # $$$$$$$ TODO [ahmad] Extend the newly added vertex to a sample with an ending time of 
            #                      the corresponding mile-stone time horizon and in the DIS

            runtimes.append(time.time()-t0)
#            print runtimes[-1]
#            print
        #========== End of the planning loop ===================================

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

    def initialize(self):
        '''Initializes the planner.'''
        initial_state = self.ts.init

        self.ts.nodes.clear()
        # FIXME: assumes that the spec does not have a disjunction as root'
        trajectory = [[initial_state], [0.0]]
        init_rosi = self.specification.rosi(trajectory)   # How to build the completions set. 
        self.ts.add_node(initial_state,
            attr_dict = {self.specification: {
#                              'costFromParent' : 0,
#                              'costFromRoot' : 0,
                             'parent': None,
                             'children': set(),
                             'trajectory': trajectory,
                             'time': 0.0,
                             'rosi': init_rosi,
                             'control': None,
                             'chi': self.chi(trajectory, self.specification, 0),
                             'ext_traj': trajectory
                            }
                        })
#         self.ts.g.best_path = deque([])
        self.ts.maxtime = 0.0

        self.lowerBoundCost = DBL_MAX
        self.lowerBoundVertex = None

    def chi(self, traj, phi, t):
        '''FIXME: inefficient, should store RoSI for subformulae
        FIXME: assumes that t corresponds to the last state in the traj TODO [Ahmad] this function is recursive 
        '''
        assert round(t,10) == round(traj[1][-1],10)

        if phi.op == STL.BOOLEAN:
            dis = np.zeros((self.system.dim))
        elif phi.op == STL.PRED:
            dis = np.zeros((self.system.dim))
            x = traj[0][-1]
            dis[phi.var_idx] = self.system.maxf(x, phi.var_idx,
                                                positive=phi.rel == STL.GREATER)
        elif phi.op == STL.NOT:
            dis = -self.chi(traj, phi.subformula, t)
        elif phi.op in (STL.OR, STL.AND):
            vec = [(self.chi(traj, f, t), f.rosi(traj, t)) for f in phi.subformulae]
            dis, rosi = vec.pop()
            for d, r in vec:
                dc, rc, dnc, _ = self.choose(dis, d, rosi, r,
                                             self.specification.space.rho_max)
                rosi = rc
                dis = blend(dc, dnc)
        elif phi.op in (STL.EVENTUALLY, STL.ALWAYS):
            dis = self.chi(traj, phi.subformula, t)
        elif phi.op == STL.UNTIL:
            raise NotImplementedError
        else:
            raise ValueError('Unknown operation, opcode: %d!', phi.op)

        assert not any(np.isnan(dis)), phi.op
        assert len(dis) == self.system.dim
        print(dis)
        return dis

    
    def ufrm_sample(self):
        '''
        Adapt the SDF and generate sample in the workspace based upon the sampled time
        '''
        t_max = min(self.ts.maxtime, self.specification.bound)
        t_rand = uniform(low=0, high=t_max) # TODO [ahmad] based on the sampled time, and the active predicate 

#       t_rand = max(0, t_max - exponential(scale=0.5))

        P_active = self.specification.active(t_rand)
        P = set([])
        while P_active:
            p1 = P_active.pop()
            for p2 in P_active:
                if p1.mutually_exclusive(p2):
                    P_active.remove(p2)
                    if bernoulli():
                        P.add(p1)
                    else:
                        P.add(p2)
                    break
            else:
                P.add(p1)

        low, high = np.array(self.system.bound).T
        for p in P:
            if p.rel == STL.LESS:
                high[p.var_idx] = p.mu
            else:
                low[p.var_idx] = p.mu
        # TODO [PY2]>>>
        # return t_rand, tuple([uniform(l, h) for l, h in it.izip(low, high)])
        # TODO [PY2]<<<
        # TODO [PY3]>>>
        return t_rand, tuple([uniform(l, h) for l, h in it.zip_longest(low, high)])
        # TODO [PY3]<<<


    
    def sample(self):
        #TODO [smplng] : Sample from the importance sampling distribution 
        '''Generates a random time and state.'''
        t_max = min(self.ts.maxtime, self.specification.bound)
        t_rand = uniform(low=0, high=t_max)
#         t_rand = max(0, t_max - exponential(scale=0.5))

        P_active = self.specification.active(t_rand)
        P = set([])
        while P_active:
            p1 = P_active.pop()
            for p2 in P_active:
                if p1.mutually_exclusive(p2):
                    P_active.remove(p2)
                    if bernoulli():
                        P.add(p1)
                    else:
                        P.add(p2)
                    break
            else:
                P.add(p1)

        low, high = np.array(self.system.bound).T
        for p in P:
            if p.rel == STL.LESS:
                high[p.var_idx] = p.mu
            else:
                low[p.var_idx] = p.mu
        return t_rand, tuple([uniform(l, h) for l, h in it.izip(low, high)])

    def near(self, x_rand, t_rand):
        '''Returns all states from the tree TS that are within the RRT* radius
        from the random sample, and within the time bound.
        ''' #TODO: add back shrinking radius
#         # Compute the ball radius
#         n = len(self.ts.nodes)
        # compute RRT* radius
        r = self.gamma#*np.power(np.log(n + 1.0)/(n + 1.0), self.system.dim)
#        print 'near radius:', r
        X_near = [x for x in self.ts.nodes if self.system.dist(x_rand, x) <= r]

        N_near = []
        for x in X_near:
            for phi, data in iter(self.ts.nodes[x].items()):
                if data['time'] + self.min_step <= t_rand \
                                                    <= data['time'] + self.T_H:
                    N_near.append((x, phi))
        return N_near

    def nearest(self, x_rand, t_rand):
        '''Returns the nearest (w.r.t. space and time) state in the RRT tree.'''
        x_pa = min(self.ts.nodes, key=lambda x: self.system.dist(x_rand, x))
        # TODO [PY2]>>>
        # times = [d['time'] for d in self.ts.nodes[x_pa].itervalues()]
        # TODO [PY2]<<< 
        # TODO [PY3]>>>
        times = [d['time'] for d in iter(self.ts.nodes[x_pa].values())]
        # TODO [PY3]<<< 
        
        t_pa_min = min(times)
        if t_pa_min + self.min_step > self.specification.bound:
            return None, None, None
        times = [t for t in times if t + self.min_step < t_rand <= t + self.T_H]

        if t_pa_min + self.min_step < t_rand:
            if times:
                t_pa_min = min(times)
            else:
                t_rand = min(t_pa_min + self.T_H, self.specification.bound)
            return x_pa, t_pa_min, t_rand

        t_rand = min(t_pa_min + uniform(self.min_step, self.T_H),
                     self.specification.bound)
        if t_rand <= t_pa_min + self.min_step:
            return None, None, None
        return x_pa, t_pa_min, t_rand

    def connect(self, x_rand, t_rand, N_near):
        '''Attempts to connect a x_rand to the best parent in N_near at time
        t_rand.
        '''
        low, high = np.array(self.system.bound).T
        #The computation of J_chi 
        lam = uniform(0, 1)
        v_pa = None
        J_star = DBL_MAX
        x_star = None
        xx_rand = np.array(x_rand)
        lx_rand = (1 - lam) * xx_rand
        traj_star = None
        u_star = None
        for xp, phip in N_near:
            assert np.all(low <= xp) and np.all(xp <= high)
            x1 = np.array(xp) + np.array(self.ts.nodes[xp][phip]['chi']) * self.T_H
            xs = lam * x1 + lx_rand 
            Js = np.sum(np.square(xx_rand - x1))
            ts = self.ts.nodes[xp][phip]['time']
            u, traj_steer, x_new = self.system.steer(xp, xs, t_rand-ts, ts) # self.system.steer(initial state, final state, starting time, ending time)
            if traj_steer:
                if np.all(low <= x_new) and np.all(x_new <= high):
                    if Js < J_star:
                        J_star = Js
                        x_star = x_new
                        v_pa = (xp, phip)
                        traj_star = traj_steer
                        u_star = u
        if x_star is None:
            return None, False
        # attempt to add to the transition system
        return self.update(v_pa, (x_star, None), t_rand, traj_star, u_star)

    def rewire(self, v_new, t_new):
        '''Attempts to rewire the RRT* tree using the new vertex, i.e., change
        the parents of nearby nodes using v_new.
        '''
        x_new, phi_new = v_new
        if x_new not in self.ts.nodes:
            return
        if phi_new not in self.ts.nodes[x_new]:
            return

        v_pa = self.ts.nodes[x_new][phi_new]['parent']
#         print 'rewire:', v_new, t_new

        N_near = self.near(x_new, t_new)
        for v in N_near:
            if v != v_new and v!= v_pa:
                x, phi_x = v
                t = self.ts.nodes[x][phi_x]['time']
                d = t_new - t
#                 print 'rewire-near:', v, t, d
                u, traj_steer, xp = self.system.steer(x_new, x, d, t)
                if traj_steer  and all(np.isclose(x, xp, rtol=0)): # exact connection
                    self.update(v_new, v, t, traj_steer, u,rewire_flg=True)

    def update(self, v_pa, v_ch, t, traj, u,rewire_flg = False):
        '''Updates the child vertex based on the potential parent vertex.
        Note: If a vertex is added to the transition system, then
        self.ts.max_time is updated.
        '''
        assert u is not None
        assert traj is not None
        assert t is not None

        x_pa, phi_pa = v_pa
        x_ch, phi_ch = v_ch

        pa_traj = self.ts.nodes[x_pa][phi_pa]['trajectory']
        ch_traj = [pa_traj[0] + traj[0][1:], pa_traj[1] + traj[1][1:]]
        
        ext_traj = []
        J_ext = 1000000
        # u, traj_steer, x_new = self.system.steer(xp, xs, t_rand-ts, ts)



#         print 'parent traj', pa_traj
#         print 'child traj', ch_traj#         print u
#         print traj[-1]


        rosi_ch = phi_pa.rosi(ch_traj, t)
        ap_ch, bp_ch = rosi_ch

        updated = False
        update_branches = False
        if phi_ch is None:
            if bp_ch > 0:
                if x_ch not in self.ts.nodes:
                    self.ts.add_node(x_ch, dict())
                phi_ch = phi_pa.simplify(ch_traj, t)
                v_ch = x_ch, phi_ch
                updated = True
        else:
            rosi_ch_prev = self.ts.nodes[x_ch][phi_ch]['rosi']
            a_ch, _ = rosi_ch_prev
            if bp_ch > 0 and ap_ch > a_ch and phi_pa.compatible(phi_ch):
                updated = True
                update_branches = True

        if updated: # update child information
            assert phi_ch is not None
            if phi_ch in self.ts.nodes[x_ch]: # rewiring
                ch_children = self.ts.nodes[x_ch][phi_ch]['children']
                # remove child vertex from its previous parent children list
                x_prev_pa, phi_prev_pa = self.ts.nodes[x_ch][phi_ch]['parent']
                self.ts.nodes[x_prev_pa][phi_prev_pa]['children'].remove(v_ch)
            else: # new node
                ch_children = set()
            # Given that the node is feasible to add to TS, we extend it for 
            # further exploration to generate the optimal sampling distribution: 
            # if rewire_flg: 
            #     ext_traj = self.ts.nodes[x_ch][phi_ch]['ext_traj']
            #     J_ext = self.ts.nodes[x_ch][phi_ch]['J_ext']
            # else: 
            #     t_du = 3
            #     t_c = t 
            #     u, ext_traj, x_new = self.system.steer_rand(x_ch, t_du, t_c) # this function assign random action given the time horizon t_h
            #     low, high = np.array(self.system.bound).T
            #     #The computation of J_chi 
            #     lam = uniform(0, 1)
            #     xx_new = np.array(x_new)
            #     lx_rand = (1 - lam) * xx_new
            #     assert np.all(low <= x_ch) and np.all(x_ch <= high)
            #     x1 = np.array(x_ch) + np.array(self.ts.nodes[x_ch][phi_ch]['chi']) * self.T_H
            #     J_ext = np.sum(np.square(xx_new - x1))
                

            self.ts.nodes[x_ch][phi_ch] = {
#                  'costFromParent' : 0,
#                  'costFromRoot' : 0,
                 'parent': v_pa,
                 'children': ch_children,
                 'trajectory': ch_traj,
                 'time': t,
                 'rosi': rosi_ch,
                 'control': u,#         print u
#         print traj[-1]
                 'ext_traj': ext_traj, 
                 'J_ext': J_ext, 
                 'chi': self.chi(traj, phi_ch, t)
                }
            # add child vertex to the new parent's children list
            self.ts.nodes[x_pa][phi_pa]['children'].add(v_ch)
            # update maxtime
            self.ts.maxtime = min(max(self.ts.maxtime, t),
                                  1.05 * self.specification.bound)
            # update best vertex
            self.check_update_best_vertex(v_ch)
            if update_branches:
                # update the cost of all vertices in the rewired branch
                self.update_branch_costs(v_ch)
#         # tree invariant
#         assert self.ts.g.number_of_nodes() == (self.ts.g.number_of_edges() + 1)
#         assert all([self.ts.g.in_degree(n)==1 for n in self.ts.g if n != self.ts.init])
        return v_ch, updated

    def check_update_best_vertex(self, v):
        '''Updates the best state that has a singleton RoSI.'''
        x, phi = v
#         print 'check best:', v
        a, b = self.ts.nodes[x][phi]['rosi']
        if np.isclose(a, b, rtol=0):
            costCurr = max(a, b)
            if self.lowerBoundVertex is None or costCurr > self.lowerBoundCost:
                self.lowerBoundVertex = v
                self.lowerBoundCost = costCurr

    def update_branch_costs(self, v):
        '''Updates the RoSI of vetices on the branches rooted at v.'''
        x, phi = v
        queue = deque([(v, self.ts.nodes[x][phi]['children'])])
        while queue:
            _, children = queue.popleft()
            # Update the cost for each children
            for child in children:
                x_ch, phi_ch = child
                traj = self.ts.nodes[x_ch][phi_ch]['trajectory']
                t = self.ts.nodes[x_ch][phi_ch]['time']
                rosi_ch = phi_ch.rosi(traj, t)
                self.ts.nodes[x_ch][phi_ch]['rosi'] = rosi_ch

                ch_children = self.ts.nodes[x_ch][phi_ch]['children']
                if ch_children:
                    queue.append((child, ch_children))
                if rosi_ch[1] < 0:
                    self.ts.nodes[x_ch].pop(phi_ch)
                    #FIXME: optimize prune

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
        # plt.fill([2, 3, 3, 2, 2], [-1.0, -1.0, -0.5, -0.5, -1.0],
        #          color=(0, 1, 0, 0.2))
        #-------------------------------------

        # plt.fill([3.5, 4, 4, 3.5, 3.5], [-0.2, -0.2, 0.2, 0.2, -0.2],
        #      color=(1, 0.5, 0, 0.2))
        # plt.fill([0.1, 1, 1, 0.1, 0.1], [0.2, 0.2, 0.5, 0.5, 0.2],
        #          color=(0, 1, 0, 0.2))

        # plt.fill([1.5, 3.25, 3.25, 1.5, 1.5], [0.5, 0.5, 1.0, 1.0, 0.5],
        #          color=(0, 1, 0, 0.2))
        # plt.fill([1.5, 3, 3, 1.5, 1.5], [-0.25, -0.25, 0.4, 0.4, -0.25],
        #          color=(.1, .1, .1, .5))
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
            filename = '../case1_di.yaml'
        elif sys.argv[1]=='rwc':
            filename = '../case2_rwc.yaml'
        else:
            raise NotImplementedError
    else:
#         filename = '../cases/case2_rwc.yaml'
        filename = '../case1_di.yaml'

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

    # load specification TODO [twtl] load the twtl specs:
    #-----------------------------------------------------------
    specification = get_specification_ast(mission.specification) 
    twtl_formula = '(H^2 x>=6) . (H^2 x<=4) . [H^2 x>=5]^[10,12]'
    # twtl_formula = '(H^2 x>=6) . (H^2 x<=4) . (H^5 x>=5)'
    # twtl_formula = '[H^2 x>=5]^[3,12]'
    lexer = twtlLexer(InputStream(twtl_formula))
    tokens = CommonTokenStream(lexer=lexer)
    parser = twtlParser(tokens)
    t = parser.formula()


    # ---------------------------------------------------------

    logging.info('Specification time bound: %f', specification.bound)

    global N
    global current_k
    global epsilon

    N = mission.planning['planning_steps']
    epsilon = 0.1
    current_k = 0
    planner = Planner(system, specification, mission.planning,mission)

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
