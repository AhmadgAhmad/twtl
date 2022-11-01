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
from ast import Del
from ftplib import parse150
import logging
import itertools as it
from re import S, T
import time
from collections import deque
import time
import timeit
import random
import numpy as np, sys
from numpy.random import uniform, exponential, randint
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm
import plotly.graph_objects as go
# import twtl.twtl  as twtl
from antlr4 import InputStream, CommonTokenStream
from twtl import translate
from twtlLexer import twtlLexer
from twtlParser import twtlParser
from twtl_ast import TWTLAbstractSyntaxTreeExtractor
from twtl_ast import Operation as Op



bernoulli = lambda p=0.5: 0 if uniform() <= p else 1
import networkx as nx

from lomap import Ts

from rhotwtl_rrt.stl import STL
from rhotwtl_rrt.hscc2017_specs import get_specification_ast
from rhotwtl_rrt.mission import Mission
from rhotwtl_rrt.systems import load_system

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


# >>>>>>>>>>>> TWTL-RRT classes <<<<<<<<<<<<<<<<<<<<<<<<
class PA(object):
    '''
    The product automaton class
    '''
    def __init__(self,p0 = None,pf = None,Sp= None,Del_p = None,Fp = None,Sa = None):
        self.Sp = Sp
        self.p0 = p0 
        self.pf = pf 
        self.Del_p = Del_p
        self.Fp = Fp 
        self.Sa = Sa # the set of automaton states 
    def smpl_nonEty_s(self): 
        '''
        sample a spec automaton state, s, such that it has a correspondence PA states, i.e., there are TS states in that region
        '''
        pass
    def smpl_s(self):
        '''
        sample a specs' automaton state
        '''
        return self.Satmtn[randint(low=0,high=len(self.Satmtn))]

    def smpl_TS_x(self):
        pass

    def sample(self,x_rand):
        '''
        
        '''
        assert x_rand is not None
        Sp = self.Sp 
        Sa = self.Sa 
        s = []
        Sps = [sa['s'] for sa in Sp ]
        while True:
            s = random.sample(Sa,1)
            if s[0] in Sps:
                break
        Vs = [xts['x'] for xts in Sp if xts['s']==s[0]] # with s being the specs automaton state:  
        # The tree of the NNs (we cannot build it outside)
        tempTree = KDTree(Vs, leaf_size=2) #TODO (meeting) build out of here (when adding a vertex to the tree)

        # The index of the NN vertex (it is different than indexID of the vertex)
        nn_currIndex = int(tempTree.query(x_rand.reshape([1, 2]), k=1, return_distance=False))
        x_exp = Vs[nn_currIndex]
        return x_exp, s[0]
    
    def update_PA(self):
        pass

    

class PAstate(object):
    '''
    Class of product automaton states 
    '''
    def __init__(self):
        self.TX_xs = None # The cooerdinates of the TS states that correspond to the automaton state. 
        self.TX_ids =None # Keep carying over the id of the vertices of intrest of TS (this will be usful when we want to retrive the actual vertices because we might need to use stored information at each node)
        self.SP = None    # Set of Paroduct automaton states
        self.TS = None
        self.Atmtn = None # the automaton that corresponds to the specifications automaton 
    def return_TS_xs(slef):
        pass

class TS(object):
    def __init__(self,x0):
        self.V = None           # the set of vertices 
        self.E = None           # the set of edges, aka transitons 
        self.dynamics = None    # the actual underlying dynamics that TS is an abstraction for. 
        self.x0  = x0           # the initial state 
        self.APs = []           # The set of APs
        
    def update(self):
        pass 
    
    def L(self,alphabet):
        '''
        This function labels the observations based off the APs
        '''
        pass
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


class Tree(object):
    '''TODO:'''
    def __init__(self):
        self.nodes = dict()
        self.init = None

    def add_node(self, state, attr_dict):
        self.nodes[state] = attr_dict


class Planner(object):
    '''Class for planner.'''

    def __init__(self, specs_ast  = [], twtl_formula = []):
        '''Constructor'''
        # self.system = system # for now we assume point dummy robot (evolvethe dynamics using linspace) TODO [code cont]: Incoroprate different types of dynamics 
        self.specification = specs_ast  # as AST
        self.twtl_formula = twtl_formula

        self.x0 = 0
        self.s0 = 0

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

        # Inspection for the twtl specs: 
        self.twtl_ast = None
        self.twtlDFA = None
        self.PA = None          # the product automaton
        self.TS = None


        # Planner params: 
        self.mission = None
        self.system = None
        self.seed  = 2000 
        self.steps = 1000
        self.gamma = 0.9
        self.save_plt_flg = False
        self.choose = choose_random
        self.T_H = 1
        self.min_step = 0.01
        assert self.min_step < self.T_H



    #$#$#$#$#$#$#$#$#$# the main planning loop: 
    def solve(self):
        '''
        Solve the motion planning problem:  
        '''
        d_steer = 1
        bounds = self.mission.system['bound']
        x_rand = np.random.uniform([bounds[0][0],bounds[1][0]],[bounds[0][1],bounds[1][1]])
        x_exp, s_rand = self.PA.sample(x_rand) # Sample a product automaton state
        traj, t_traj = self.system.steer(x0 = x_exp, xd= x_rand, d_steer = d_steer, ti = 0)
        x_new = traj[-1,:]
        # finding the near set: 
        Vnear = self.near(s = s_rand, x_new = x_new, d_steer = d_steer)
        x_max = x_exp
        x_max = self.bestParent(x_max,Vnear)
        # Might not need steering here 
        a = 1 




        #=================================================================
        #=================================================================
        #=================================================================


    def initialize(self, twtl_formula = None, x0 = None, s0 = None):
     
        # Translate the twtl formula into a DFA: 
        assert twtl_formula is not None
        lexer = twtlLexer(InputStream(twtl_formula))
        tokens = CommonTokenStream(lexer=lexer)
        parser = twtlParser(tokens)
        phi = parser.formula()
        twtl_ast =  TWTLAbstractSyntaxTreeExtractor().visit(phi)
        DFAresult = translate(ast=twtl_ast,norm=True)
        self.DFAphi = DFAresult[1] # The determenstic finite automaton of the twtl specifications  
        # Instantiate the transition system (will be built incrementally); essentially the RRT* tree: 
        self.TS = TS(x0 = x0) 
         
        # Instantiate the product automaton: 
        p0 = {'s': self.DFAphi.init[0], 'x': self.TS.x0}
        pf = {'s': self.DFAphi.final.pop(), 'x': self.TS.x0}
        Sp = [p0]
        Del_p = []
        Fp = [pf]
        self.PA = PA(p0 = p0, pf = pf, Sp= Sp, Del_p= Del_p, Fp = Fp,Sa=self.DFAphi.g.nodes())
        # environment instantiation: 
        filename = '/home/ahmad/Desktop/twtl/twtl/case1_point_robot.yaml'
        mission = Mission.from_file(filename)
        # set seed
        np.random.seed(mission.planning['seed'])

        # load system model
        system = load_system(mission.system)
        self.mission = mission 
        self.system = system

    
    

    # >>>>>>>>>>>>>>>>TWTL-based primitives functions <<<<<<<<<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    def near(self, s, x_new, d_steer):
        """
        Find all the vertices within a  ball with radius=b_raduis of the Vertex. Uses KDball-trees as well.
        :param b_raduis: a ball in which we search for the neighboring vertices
        :input x_new:   a state in X_free (NOT a vertex yet, we got it after steering x_exp)
        :input s_nnEty: a twtlAutomaton state that has a corresponding PA state/s
        :return: A list of the vertexes with the specified ball.
        """
        assert x_new is not None
        Sp = self.PA.Sp 
        Vs = [xts['x'] for xts in Sp if xts['s']==s] # with s being the specs automaton state:  
        # The tree of the NNs (we cannot build it outside)
        tempTree = BallTree(Vs)
        # The index of the NN vertex (it is different than indexID of the vertex)
        nn_currIndex, nn_currDist = tempTree.query_radius(x_new.reshape([1, 2]), r=d_steer, return_distance=True,sort_results=True) 
        V_near = [Vs[i] for i in nn_currIndex[0]]
        return V_near


    def V_next(self,s,v_new):
        '''
        This function returns the a set of states in from V, the set of TS states, 
        : s : the current automaton state
        : v_new : the TS state from which we want to find the next automaton state, s_next, that if s_next.TX_xs 
        is not empty, we'll try to rewire those states based on the ribustness  
        :
        '''

        # Steps: 
        
        # 1) Find s_next following the tranisiton: \delta(s,l(v_new)) 

        # 2) Given s_next, v_new, and d_steer find the near set: 
        return self.near(b_radi=self.d_steer,x_new=v_new.s,s_nnEty=s)


    def bestParent(self,s_rand,x_max,x_new,Vnear):
        
        for x in Vnear:
            trajp, t_traj = self.system.steer(x0 = x, xd = x_new, exct_flg = True)
            xp = trajp[-1,:]   
            if np.linalg.norm(xp-x_new)<0.05: # TODO [crucial] compute the robustness or what ever 
                x_max = x
                Lx_max = self.TS.L(x_max) # This function should check to what Region/Property/Proposition the state x_max belongs
                s_max = self.DFAphi.next(s_rand,Lx_max) # Basically this function will check the next specification automaton state 
                p_new = {'s':s_max,'x':x_max}
                self.PA.Sp.append(p_new)
                # Do we need to update the transition system per say? 


                

    
    def find_vmax_rhp(self,x_max,v_new,V_near,s_rand):
        '''
        This is basically choosing the best parent
        '''
        # TODO [build code] Make sure to distinguish between x and v  
        for v in V_near: 
            x_prime, traj = self.steer( x_start = v, x_g = v_new, exct_flg = True)
            if abs(x_prime-v_new)< self.eps and self.collFree(traj) and self.computeCost(v) >= self.computeCost(x_max):
                p_new = 1 
                x_max = v
            # TODO [code build] Update the PA states as well as the transitions. $(subsequently, update the transion system as well s_rand.TX_xs 
            # (the set of TS states that corresponds to PA states that their s state variable is s_rand))$
    def rewire(self,V_next, x_new):
        '''
        Check if the rewiring the TS vertices in the next specifications level 
        (PA states with s = next automaton state), would increase the robust satisfaction 
        $$$ >>>>>>
        Let's for now use the Eucledean distance as our the cost fucntion that we want to optimize. 
        This will help in debugging the RRT with exploring the product automaton statespace 
        <<<<<<< $$$

        '''
        for v in V_next: 
            
            pass
        pass
    
    def steer(self,t_i,x_start = None, x_g = None, d_steer = None, exct_flg = False):
        '''
        assume dummy connections are achievable between the states (i.e. all the states are reachable from each other)
        TODO [code build] consider the following dynamics: single and double integrators, bicycle and unicycle models 
        (maybe enclode extended models in which the acceleration is the input control)
        : param d_steer:    If given a value and exct_flg is False, then we aim to steer the system from x_start toward x_g to a state that is d_steer distance away from x_start
        : param exct_flg:   If True, then we aim to steer the system from x_start to x_g (note: x_g might not be reachable from x_start)
        : input x_start: an initial system state
        : input x_g: a state that we want to steer to or to steer towards. 
        > return traj, t_traj
        '''
        if exct_flg is True and x_g is not None:
            traj = np.linspace(start=x_start,stop=x_g,num=numSteps)
        elif not exct_flg:
            x_g = 1#compute a state in the direction of x_g and d_steer distaned 
        pass
    # Quantitative semantics of TWTL  
    def computeCost(self):
        pass
    def robustness(self):
        pass
    def rtimeMntrng(self):
        pass


    # >>>>>>>>>>>>>>>>End of TWTL-based primitives functions <<<<<<<<<<
    # >>>>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


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
    
    
    #XXX
    #-------------
    # Planner Class instentiation: 
    planner = Planner() 
    # Initialize the planning algorithm (Line 3 - 4):
    x0 = (0,0)
    s0 = 0
    twtl_formula = 'H^3 A . H^5 B . [H^5 A]^[10,15]'
    twtl_formula = 'H^3 A'
    twtl_formulaPred =  'H^3 (x1>1 && x1<2 && x2>1 && x2<3) . H^5 (x1>2 && x1<4 && x2>1 && x2<3) . [H^5 (x1>1 && x1<2 && x2>1 && x2<3)]^[10,15]'
    
    planner.initialize(twtl_formula,x0 = x0, s0 = s0)
    # Solve the planning problem (Line 6 - 30):   
    planner.solve()
    #-------------
    #XXX



    # twtl_formula = '(H^2 x>=6) . (H^2 x<=4) . [H^2 x>=5]^[10,12]'

    twtl_formula = '[H^3 x1>2 &&  H^3 x1<4 && H^3 x2>2 && H^3 x2<4]^[5,10]'
    twtl_formula = '[H^3 A &&  H^3 B && H^3 C && H^3 C]^[5,10]'
    twtl_formula = 'H^3 x1>2'
    twtl_formula = 'H^3 A . H^5 B . [H^5 A]^[10,15]'
    twtl_formula  = 'H^3 x<3 . H^4 A . H^7 x<3'
    # twtl_formula = '(H^2 x>=6) . (H^2 x<=4) . (H^5 x>=5)'
    # twtl_formula = '[H^2 x>=5]^[3,12]'
    
    # >>> Define the  specs using APs as well as linear    
    twtl_formula = 'H^3 A . H^5 B . [H^5 A]^[10,15]'
    A = 'x1>2 && x1<3 && x2>2 && x2<3' 
    B = 'x1>4 && x2<3'

    lexer = twtlLexer(InputStream(twtl_formula))
    tokens = CommonTokenStream(lexer=lexer)
    parser = twtlParser(tokens)
    phi = parser.formula()

    twtl_ast =  TWTLAbstractSyntaxTreeExtractor().visit(phi)
    twtlDFA = translate(ast=twtl_ast ,norm=True)
    lexer.VARIABLE
    # ---------------------------------------------------------

    # logging.info('Specification time bound: %f', specification.bound)

    global N
    global current_k
    global epsilon

    epsilon = 0.1
    current_k = 0
    planner = Planner(specs_ast=twtl_ast,twtl_formula = phi)    

    np.random.seed(planner.seed)
    twtlDFA = translate(ast=twtl_ast ,norm=True)
    Planner.twtlDFA = twtlDFA
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


    logging.info('Done!')

if __name__ == '__main__':
     main()
