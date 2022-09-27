'''
Created on Aug 20, 2016

@author: cristi
'''

import itertools as it

import numpy as np
from scipy.integrate import odeint
from scipy.optimize import *
import math
import logging
from rhotwtl_rrt.mission import Mission
import matplotlib.pyplot as plt


def load_system(params):
    '''Loads a system with the given parameters.'''
    if params['type'] == 'double_integrator':
        return DoubleIntegrator(params)
    elif params['type'] == 'rear_wheel_car':
        return RearWheelCar(params)
    raise ValueError('Unknown system %s!', str(params['type']))


class DoubleIntegrator(object):
    '''Double integrator system \ddot{x} = u .'''
    # TODO [case_dyn] This case study plot the acceleration velocities directly,
    # TODO [ahmad_q] The formulation of the double integrator dynamics is wrong in the paper.  
    
    def __init__(self, params):
        '''Constructor'''
        assert params['type'] == 'double_integrator'
        
        self.bound = tuple([tuple(map(float, b)) for b in params['bound']])
        self.control_bounds = tuple([float(b) for b in params['control_bounds']])
        self.dim = len(self.bound)
        self.dt = float(params['dt'])
        
        self.initial_state = tuple(params['initial_state'])
        assert len(self.initial_state) == self.dim
        # TODO [PY2]>>>
        # assert all([l <= x <= h for (l, h), x in
        #                               it.izip(self.bound, self.initial_state)])
        # TODO [PY2]<<< 
        # TODO [PY3]>>>
        assert all([l <= x <= h for (l, h), x in
                                      it.zip_longest(self.bound, self.initial_state)])
        # TODO [PY3]<<< 

    
    def steer(self, a, b, d, ti, u=None, eta=0.2):
        #(initial_state, final_state, starting_time, ending_time)
        '''Steering primitive.'''
        assert d >= 0.001
#         print 'steer:', a, b, d, eta
        
        aa = np.asarray(a)
        bb = np.asarray(b)
        diff = bb - aa
        dist = np.linalg.norm(diff)
        if dist > eta:
            bb = aa + eta * diff /dist
        u = (bb[1] - aa[1]) / d # TODO [ahmad_q] Why  the control input is assigned as such? Also, since this is a point robot open loop control inputs are computed easily. 
#         u = 2 *(bb[0] - aa[0] - aa[1] * d) / (d**2)
#         bb[1] = aa[1] + u * d
#         
#         print 'steer:', u
        t = np.linspace(0, d, num=int(np.ceil(d/self.dt)+1))
        # TODO [PY2]>>>
        # traj = zip(aa[0] + aa[1] * t + u * t*t / 2, aa[1] + u * t)
        # TODO [PY2]<<<
        # TODO [PY3]>>>
        traj = list(zip(aa[0] + aa[1] * t + u * t*t / 2, aa[1] + u * t))
        # TODO [PY3]<<<

        assert len(traj) >= 2
        assert len(traj) == len(t)
#         print traj
#         print t
        return u, (traj, list(ti + t)), traj[-1]
    
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

    def steer_undrU(self,x0,ti,u_seq):
        '''
        Given an open loop control inputs, u_ol, with the corres[ponding time traj, evolve the system and compute the J_chi.
        Evaluate the elite set based on J_chi of the ending location of the generated trajectory: 
        - u_ol   : Randomly generated control inputs 
        - t_traj : time trajectory (computed based on sub-time horizons)
        - x_0    : The state of the vertex we are extending 
        Note: This function differes from CE-RRT* in a since that we don't extend to the goal, becayuse we don't have a well defined goal, 
        rather, we have STL specs. 
        As a starting trial, we evalute the elite set based upon J_chi of the latest added vertex of the random sample. 
        '''
        nt = len(u_seq)
        dt = self.dt
        t = np.linspace(dt, nt*dt+dt, num=int(nt))
        traj = []
        x1_0 = x0[0]
        x2_0 = x0[1]
        for u in u_seq: 
            x1_1 = x1_0 + x2_0 * dt
            x2_1 = x2_0 + u * dt  
            x = (x1_1,x2_1)     
            traj.append(x)
            x1_0 = x1_1
            x2_0 = x2_1
        return u_seq, (traj, list(ti + t)), traj[-1]
    
    def get_x_dot(self):
        pass
    
    def dist(self, a, b):
        '''Returns the Euclidean distance between the two states.'''
        dx = a[0]-b[0]
        dy = a[1]-b[1]
        return np.sqrt(dx*dx+dy*dy)
    
    def maxf(self, x, i, positive):
        '''Returns the value of f_i that maximizes \abs{f_i} over all u such
        that f_i > 0 if positive is true, otherwise f_i < 0.
        '''
        assert 0 <= i < self.dim
        if i == 0:
            if positive:
                return max(x[1], 0)
            else:
                return min(x[1], 0)
        else:
            if positive:
                return self.control_bounds[1]
            else:
                return self.control_bounds[0]


class RearWheelCar(object):
    '''Rear wheel car model
    '''
    
    def __init__(self, params):
        assert params['type'] == 'rear_wheel_car'
        
        self.bound = tuple([tuple(map(float, b)) for b in params['bound']])
        self.control_bounds = tuple([tuple(map(float, b))
                                            for b in params['control_bounds']])
        self.dim = len(self.bound)
        self.dt = float(params['dt'])
        
        self.initial_state = tuple(params['initial_state'])
        assert len(self.initial_state) == self.dim
        assert all([l <= x <= h for (l, h), x in
                                      it.izip(self.bound, self.initial_state)])
    
    def steer(self, a, b, d, ti, eta=0.2):
        '''TODO: 
        '''
        #TODO [ahmad_q] what if b is not reachable from a? 
#         print 'steer:', a, b
#         assert d >= 0.001
# #         print 'steer:', a, b, d, eta
#         
#         aa = np.asarray(a)
#         bb = np.asarray(b)
#         diff = bb[:2] - aa[:2]
#         dist = np.linalg.norm(diff)
#         if dist > eta:
#             bb[:2] = aa[:2] + eta * diff /dist
        
        
        def f(x, t, u):
            dx = np.zeros_like(x)
            dx[0] = x[3] * np.cos(x[2])
            dx[1] = x[3] * np.sin(x[2])
            dx[2] = x[4]
            dx[3] = u[0]
            dx[4] = u[1]
            return dx
        
        def opt_control(u):
            traj = odeint(f, a, [ti, ti+d], args=(u,))
            x_term = traj[-1]
            return self.dist(x_term, b)
        
        res = minimize(opt_control, [0.0, 0.0], bounds=self.control_bounds,
                       method='TNC')
        
        u = tuple(res.x)
        t = ti + np.linspace(0, d, num=np.ceil(d/self.dt)+1)
        traj = map(tuple, odeint(f, a, t, args=(u,)))
        
        return u, (list(traj), list(t)), traj[-1]
    
    def dist(self, a, b):
        '''TODO:
        Returns the distance between two states as the Euclidean distance
        between their locations.
        '''
        dx = a[0]-b[0]
        dy = a[1]-b[1]
        dtheta = a[2]-b[2]
        dv = a[3]-b[3]
        domega = a[4]-b[4]
        return np.sqrt(dx*dx + dy*dy + dtheta*dtheta + dv*dv + domega*domega)
    
    def maxf(self, x, i, positive):
        '''Returns the value of f_i that maximizes \abs{f_i} over all u such
        that f_i > 0 if positiuve is true, otherwise f_i < 0.
        '''
        assert 0 <= i < self.dim
        if positive:
            pr = max
        else:
            pr = min
        
        if i == 0:
            return pr(x[3]*np.cos(x[2]), 0)
        elif i==1:
            return pr(x[3]*np.sin(x[2]), 0)
        elif i==2:
            return pr(x[4], 0)
        elif i==3:
            return pr(self.control_bounds[0])
        else:
            return pr(self.control_bounds[1])


if __name__ == '__main__':
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
    Td = 15 
    n_s=10
    x0 = [0,0]
    for i in range(200):
        u_seq = system.sample_us(Td=Td,n_s=n_s)
        u, traj,xf = system.steer_undrU(x0 = x0,ti = 1.5,u_seq=u_seq)
        traj_state = traj[0]
        traj_state = np.array(traj_state)
        plt.plot(traj_state[:,0],traj_state[:,1])
    plt.savefig('DintegTd%dn_s%d_x0_%f_%f.png' % (Td,n_s,x0[0],x0[1]))
    a = 1



    #Testing for rear_wheel_car
    # params = {
    #     'type': 'rear_wheel_car',
    #     'initial_state': [0.1, 0.2, 0.4, 0, 0],
    #     'bound': [[0, 4], [0, 4], [-3.14, 3.14], [-0.3, 0.3], [-1, 1]],
    #     'control_bounds': [[-0.05, 0.05], [-0.2, 0.2]],
    #     'dt': 0.1,
    # }
    
    # system = RearWheelCar(params)
    
    # u, traj, x = system.steer(params['initial_state'],
    #                           [10, 15, 0.8, 0.6, 1], 10, 1)
    # print(x)
    # print(system.w_dist(params['initial_state'], [2, 4, 0, 0.01, 0.2]))
    # print(system.w_dist(x, [2, 4, 0, 0.01, 0.2]))
