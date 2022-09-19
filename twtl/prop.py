'''
Define a class of Atomic propositions based of linear-predicates 
@ Ahmad Ahmad (ahmadgh@bu.edu) Hybrid and Networked Systems (HyNeSs) Group, 
BU Robotics Lab, Boston University
'''


from twtl import * 



class AP(object):
    def __init__(self,
                h = None, 
                inst_rho = None,
                L = None):
        
        self.h = h           # The h-function (typically linear function), defined over the observed states 
        self.inst_rho = inst_rho    # The instantaneous robustness of the AP 
        self.L = L           # The labeling function, defined over the observations 


        

    def get_h_value(self,t = None, ot = None, o_traj = None):
        '''
        Computes the base case of computing the 
        '''
        pass

        
