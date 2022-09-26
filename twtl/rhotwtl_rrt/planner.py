'''
Created on Aug 20, 2016

@author: cristi
'''

import logging
import itertools as it
import time
from collections import deque

import numpy as np
from numpy.random import uniform, exponential
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

    def solve(self):
        '''
        TODO: - Inspect every newlly added edge to the tree, 
        '''
        self.initialize()

        runtimes = []
        
        # =========== The planning loop (Line 3 - 18 Algorithm 1) ================
        for k in range(self.steps):
            current_k = k
            if not k%100:
                logging.info('step: %d %d', k, len(self.ts.nodes))
            t0 = time.time()

#             print '----------------------------------'

            t_rand, x_rand = self.sample()
            # t_rand, x_rand = self.IS_sample()

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
            v_new, connected = self.connect(x_rand, t_rand, N_near)

#             print v_new
#             print connected
            # connected = False
            if connected:
                self.rewire(v_new, t_rand)

            # $$$$$$$ TODO [ahmad] Inspect the tree 
            if len(self.ts.nodes) % 100 == 0: #plot the tree every 10 iterations: 
                debug_show_ts(self, self.mission.system['type'])
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
        init_rosi = self.specification.rosi(trajectory)
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
                             'chi': self.chi(trajectory, self.specification, 0)
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
        return dis

    
    def IS_sample(self):
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
        return t_rand, tuple([uniform(l, h) for l, h in it.izip(low, high)])



    
    def sample(self):
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
            for phi, data in self.ts.nodes[x].iteritems():
                if data['time'] + self.min_step <= t_rand \
                                                    <= data['time'] + self.T_H:
                    N_near.append((x, phi))
        return N_near

    def nearest(self, x_rand, t_rand):
        '''Returns the nearest (w.r.t. space and time) state in the RRT tree.'''
        x_pa = min(self.ts.nodes, key=lambda x: self.system.dist(x_rand, x))
        times = [d['time'] for d in self.ts.nodes[x_pa].itervalues()]
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
            u, traj_steer, x_new = self.system.steer(xp, xs, t_rand-ts, ts)
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
                    self.update(v_new, v, t, traj_steer, u)

    def update(self, v_pa, v_ch, t, traj, u):
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
    import matplotlib.pyplot as plt

    figure = plt.figure()
    figure.add_subplot('111')
#     axes = figure.axes[0]
#     axes.axis('equal') # sets aspect ration to 1

    plt.xlim(planner.system.bound[0])
    plt.ylim(planner.system.bound[1])

    # draw regions
    if system_type == 'double_integrator':
        x, y = zip(*planner.ts.nodes)
        #$$$ Cristi's Specs: 
        plt.fill([3.5, 4, 4, 3.5, 3.5], [-0.2, -0.2, 0.2, 0.2, -0.2],
             color=(1, 0.5, 0, 0.2))
        plt.fill([0, 2.1, 2.1, 0, 0], [-0.5, -0.5, 0.5, 0.5, -0.5],
                 color=(0, 0, 1, 0.2))

        plt.fill([2, 3, 3, 2, 2], [0.5, 0.5, 1.0, 1.0, 0.5],
                 color=(0, 1, 0, 0.2))
        plt.fill([2, 3, 3, 2, 2], [-1.0, -1.0, -0.5, -0.5, -1.0],
                 color=(0, 1, 0, 0.2))
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
        for phi in planner.ts.nodes[x]:
            traj, _ = planner.ts.nodes[x][phi]['trajectory']
            if system_type == 'double_integrator':
                xx, yy = zip(*traj)
            elif system_type == 'rear_wheel_car':
                xx, yy,_, _, _ = zip(*traj)
            plt.plot(xx, yy, 'k')
            # Plot DIS for each node: 
            u,v = planner.ts.nodes[x][phi]['chi']
            plt.quiver(x[0],x[1],u,v,width = 0.001,color = 'g')

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
