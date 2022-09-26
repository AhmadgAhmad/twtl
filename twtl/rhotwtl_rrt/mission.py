'''
Created on Dec 4, 2015

@author: cristi
'''

import os
import itertools as it
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from yaml import load as load, YAMLObject
from yaml import Loader
import yaml



class Mission(YAMLObject):
    '''Class used to store mission data.'''
    
    def __init__(self):
        self.name = ''
        self.description = ''
        self.environment = None
        self.specification = None
        self.no_active_vehicles = 0
        self.vehicle_size = 0
        self.vehicles = None
    
    def __str__(self):
        return """\
Name: {name}
Description: {descr}
Environment: {bases} {sites}
Specification: {spec}
No vehicle: {nvehicles}
Timestep: {timestep}

Environment: {env}
Vehicles: {vehicles}
""".format(name=self.name, descr=self.description,
                   bases=self.environment['bases'].keys(),
                   sites=self.environment['regions'].keys(),
                   spec=self.specification,
                   nvehicles=self.no_active_vehicles,
                   timestep=self.timestep,
                   env=self.environment,
                   vehicles=self.vehicles)
    
    @classmethod
    def from_file(cls, filename):
        # Ahmad's Ubunto location: 
        filename = '/home/ahmad/Desktop/twtl/twtl/rhotwtl_rrt/case1_di.yaml'
        
        # Ahmad's windows location: 
        # filename = 'C:/Users/ahmad/Documents/Graduate_study/PhD_work/IS_STL_plnng/imprtnc_smplg_stl_plng/rho-rrt/cases/case1_di.yaml'
        
        with open(filename, 'r') as fin:
            return yaml.load(fin,Loader=Loader) #load(fin)
    
    def draw_ts(self, ts, viewport, figure):
        if not ts.g.number_of_nodes():
            return
        # draw nodes
        x, y = zip(*[s.conf[:2] for s in ts.g.nodes_iter()])
        viewport.plot(x, y, 'o', color='k') # TODO: node color
        for c, cov in [(s.conf[:2], s.cov[:2, :2]) for s in ts.g.nodes_iter()]:
            if np.trace(cov) <= 10:
                viewport.add_patch(Mission.covariance_ellipse(c,cov, color='b',
                                                    fill=False, lw=1, zorder=2))
#         for s, d in ts.g.nodes_iter(data=True):
#             x, y = s.conf[:2]
#             viewport.text(x+ 0.05, y + 0.05,str(d['id']), fontsize=16,
#                         horizontalalignment='center', verticalalignment='center',
#                         color='k')
        
        # draw transitions
        for start, stop, d in ts.g.edges_iter(data=True):
            if start != stop:
                color = d.get('color', 'k') #TODO: edge color
                path = [start, stop] # TODO: extract nominal path
                line_style = d.get('line_style', '-') # TODO: line style
                for u, v in it.izip(path[:-1], path[1:]):
                    x, y = np.array([u.conf[:2], v.conf[:2]]).T
#                     print
#                     print u
#                     print v
#                     print x, y
#                     print
                    viewport.plot(x, y, color=color, linestyle=line_style)
#                     assert 0.05 <= np.sqrt(np.diff(x)**2 + np.diff(y)**2) <=0.3, np.sqrt(np.diff(x)**2 + np.diff(y)**2)
#                 if d.has_key('label_position'):
#                     x, y, _ = np.mean([start, stop], axis=0)
#                     dx, dy = e['label_position']
#                     viewport.text(x+ dx, y + dy, str(e['duration']), fontsize=16,
#                               horizontalalignment='center', verticalalignment='center',
#                               color=color)
#     
#     def draw_time_label(self, time, loop, d_fsa):
#         '''Draws the time and loop label in the current figure's title.'''
#         plt.title('Step:{: 3d} Time: {: 3d} Loop: {: 2d}   FSA distance to final: {: 4d}'
#                   .format(time, time*self.timestep, loop, d_fsa))
    
    def visualize(self, ts, plot='none', outputdir=''):
        '''TODO: Creates a video or a series images from the vehicles' trajectories.
        '''
        if plot == 'none':
            return
        if plot not in ['figures', 'video', 'design', 'plot']:
            raise ValueError('Expected "none", "figures" or "video", got {}!'.format(plot))
         
        figure = plt.figure()
        figure.add_subplot('111')
        axes = figure.axes[0]
        axes.axis('equal') # sets aspect ration to 1
          
        b = np.array(self.environment['boundary']['limits'])
        plt.xlim(b.T[0])
        plt.ylim(b.T[1])
          
        self.draw_environment(axes, figure)
        if plot == 'design':
            plt.show()
            return
        
        if plot == 'plot':
#             vehicle = next(self.vehicles.itervalues())
#             # draw vehicle
#             v, r = vehicle['initial_state'][:2], vehicle['size']
#             c = plt.Circle(v, r, color=vehicle['color'], fill=True)
#             figure.gca().add_artist(c)
#             cov = np.array(vehicle['initial_covariance'])[:2, :2]
#             axes.add_artist(self.covariance_ellipse(v, cov,
#                             color=vehicle['color'], fill=False, lw=1, zorder=2))
            # draw belief transition system
            self.draw_ts(ts, axes, figure)
            plt.show()
            return
        
         
#         def init_anim():
#             for vehicle in vehicles.values():
#                 vehicle.set_state('Base', vehicle.fuel_model['capacity'], Vehicle.np)
#                 vehicle.draw()
#                 vehicle.draw_state_label()
#             return []
#          
#         def run_anim(frame, *args):
#             logging.info('Processing frame %s!', frame)
#             self.draw_time_label(frame, loops_iter.next(), d_fsa.next())
#             for vehicle in vehicles.values():
#                 try:
#                     state, fuel, _, mode = vehicle.trajectory_iter.next()
#                     vehicle.set_state(state, fuel, mode)
#                     vehicle.draw()
#                     vehicle.draw_state_label()
#                 except StopIteration:
#                     vehicle.set_state('Base', vehicle.fuel_model['capacity'], Vehicle.np)
#                     vehicle.draw()
#             return []
#           
#         N = len(vehicles.values()[0].trajectory)
#         if plot == 'figures':
#             fname = os.path.join(outputdir,
#                                 self.simulation['figures']['filename_template'])
#             for frame in range(N): # save frames
#                 run_anim(frame)
#                 figure.savefig(fname.format(frame=frame))
#         elif plot == 'video':
#             vehicle_animation = animation.FuncAnimation(figure, run_anim,
#                                 init_func=init_anim, save_count=N,
#                                 interval=self.simulation['video']['interval'],
#                                 blit=False)
#             filename = os.path.join(outputdir, self.simulation['video']['file'])
#             logging.info('Saving video to file: %s !', filename)
#             vehicle_animation.save(filename, metadata={'artist':
#                                                        'Cristian-Ioan Vasile'})
#           

if __name__ == '__main__':
    pass