
import normflows
import torch
from torch.distributions.multivariate_normal import MultivariateNormal as mvnrnd 

import numpy as np
from numpy.random import uniform, exponential
from sklearn.neighbors import KDTree
from sklearn.neighbors import BallTree
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import cm
import plotly.graph_objects as go

class CEM(object):

    def __init__(self, est_type = 'GMM', dim = 2) -> None:
        self.est_type = est_type # 'GMM', 'KDE', 'NFs'

        # Data attributes: 
        self.dim = dim 
        # GMM params: 
        self.mu = torch.zeros(dim)
        self.cov = torch.eye(dim)
        if est_type is 'GMM':
            self.sdf = mvnrnd(self.mu, self.cov)  # will be defined based on the sampling density function. 
        else: 
            self.sdf = None # TODO [smplng] define the other samping distributions 



    def update_gmm_params(self, mu = None, cov = None):
        '''
        Compute the updating step of the mean, mu, and the coveriance matrix. See 
        https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.901.5211&rep=rep1&type=pdf page 8 (left column) parameters update.
        '''
        # TODO [code_opt] you might not need a method just to update the sdf params 
        assert mu != None
        assert cov != None
        self.mu = mu
        self.cov = cov
        # self.sdf.covariance_matrix = cov
        # self.sdf.mu = mu 
        self.sdf = mvnrnd(mu, cov) # TODO [code_opt_im] hack changing the mean.

    def sample(self,n_smpls = 1):
        '''
        Sampling a number of samples from GMM
        @ input: n_smpls: the number of samples with self.dim dimnsion 
        @ return: smpls: n_smpls samples drown from self.sdf (a GMM)  
        '''
        smpls = torch.zeros(self.dim,n_smpls) # TODO [code_opt] This could be an attribute that I could just grab whenever is needed
        for i in range(n_smpls):
                smpl = self.sdf.sample() # Make a cem_mvrnd class 
                smpls[:,i] = smpl 
        return smpls

