#!/usr/bin/env python3
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from util import add_angles, angled_vector, clip_angle, unit_vector
from segmentation import get_wall_influence

class WallModel:
    """This is an implementation of the wall model of Calovi et al. disentangling...
    https://doi.org/10.1371/journal.pcbi.1005933
    We support different expansion for the odd function.
    """
    angular_map = {'calovi': lambda angle, p1: np.sin(angle) * (1 + p1 * np.cos(2*angle)),
                   'sin': lambda angle, p1, p2, p3: p1 * np.sin(angle) + p2 * np.sin( 2 * angle ) + p3 * np.sin( 3* angle),
                   'sin-cos': lambda angle, a1, a2, b1, b2 : (a1 * np.sin(angle) + a2 * np.sin(2*angle)) * \
                   (b1 * np.cos(angle) + b2 * np.cos(2*angle) + 1),
                   'shifted-sin-cos': lambda angle, a1, a2, b1, b2:
                   (a1 * np.cos(angle) + a2 * np.sin(2 * (angle + np.pi/2))) *
                   (-b1 * np.sin(angle) + b2 * np.cos(2 * (angle + np.pi/2))+ 1)}
    params_map = {'calovi': np.array([1, 6.0, 0.7]),
                  'sin':  np.array([1, 6.0, 0.7, 0.0, 0.0]),
                  'sin-cos': np.array([1, 6.0, 0.7, 0.0, 0.0, 0.0]),
                  'shifted-sin-cos': np.array([1, 6.0, 0.7, 0.0, 0.0, 0.0])}
    
    def __init__(self, angular_model='sin-cos'):
        self.angular_model = angular_model
        self.params = self.params_map[self.angular_model]

        # Unfitted versions
        self._wall_force = lambda dist, decay: np.exp(-(dist/decay)**2 )
        self._wall_repulsion = self.angular_map[self.angular_model]

        # Initialize fitted versions.
        self.set_params(self.params)
    
    def evaluate_raw(self, xdata, *params):
        """Evaluate model with xdata, where xdata is an array with entries [wall_distance * 4, wall_angle * 4].
        """
        self.set_params(np.array(params))
        return self.__call__(xdata[0:4], xdata[4:])
    
    def set_params(self, params):
        """Set params from a vector. This is needed for scipy's curvefit function.
        The first entry is a constant multiplicator, the second the decay term and the rest are parameters of the odd function."""
        self.params = params
        self.wall_force = lambda dist: self._wall_force(dist, params[1])
        self.wall_repulsion = lambda angle: self._wall_repulsion(angle, *params[2:])
        self.scale = params[0]
        
    def __call__(self, radius, angle):
        """Return predicted heading change for radius radius and angle angle."""
        # Only consider the two closest walls.
        num = 2
        idx_rows = np.argsort(radius, axis=0)[:num, :]
        idx_cols = np.arange(radius.shape[1])[None, :]
        radius = radius[idx_rows, idx_cols]
        angle = angle[idx_rows, idx_cols] 

        # Same parameters are used for all four walls.
        return np.sum(self.scale * self.wall_force(radius) * self.wall_repulsion(angle), axis=0)

def even_fun(angle, *weights):
    """Cos series for even functions."""
    result = 1.0
    for i, weight in enumerate(weights):
        result += weight * np.cos((i+1) * angle)
    return result

def odd_fun(angle, *weights):
    """Sin series for odd functions."""
    result = 0.0
    for i, weight in enumerate(weights):
        result += weight * np.sin((i+1) * angle)
    return result

class SocialModel:
    """This is an implementation of the social model of Calovi et al. disentangling...
    https://doi.org/10.1371/journal.pcbi.1005933
    paper.
    """
    def __init__(self, num_fourier=2):
        self.num_fourier = num_fourier
        # Note: If you intend on using this model. you probably should change
        # or randomize the initial parameters.
        params = np.hstack((np.array([0.3, 2.0,1.0]), np.array([1.0] * 3),
                            np.zeros(num_fourier * 4)))
        # Raw functions
        self._f_att = lambda dist, p1, p2, s: s * (dist - p1)/(1 + (dist/p2)**2)
        self._f_ali = lambda dist, p1, p2, s: s * (dist + p1) * np.exp(-(dist/p2)**2)
        
        # Initialize fitted versions.
        self.set_params(params)
                                
    def evaluate_raw(self, xdata, *params):
       self.set_params(np.array(params)) 
       return self.__call__(xdata[0], xdata[1], xdata[2])
    
    def set_params(self, params):
        self.params = params

        # Attraction
        self.f_att = lambda dist: self._f_att(dist, *self.params[0:3])
        self.o_att = lambda a: odd_fun(a, *self._get_params_slice(0))
        self.e_att = lambda a: even_fun(a, *self._get_params_slice(1))

        # Alignment
        self.f_ali = lambda dist: self._f_ali(dist, *self.params[3:6])
        self.o_ali = lambda a: odd_fun(a, *self._get_params_slice(2))
        self.e_ali = lambda a: even_fun(a, *self._get_params_slice(3))
        
    def __call__(self, distance, viewing_angle, relative_angle):
        attraction = self.f_att(distance) * self.o_att(viewing_angle) * \
                     self.e_att(relative_angle)
        alignment = self.f_ali(distance) * self.o_ali(relative_angle) * \
                    self.e_ali(viewing_angle)
        return attraction + alignment

    def _get_params_slice(self, num_fun):
        offset = 6 + num_fun * self.num_fourier
        return self.params[offset:offset+self.num_fourier]
