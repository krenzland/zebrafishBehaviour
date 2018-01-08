#!/usr/bin/env python3
import dill as pickle
import numpy as np

from util import add_angles, angled_vector, clip_angle, unit_vector
from segmentation import get_wall_influence

def unit_vector(v):
    '''Returns a unit vector.'''
    return v/np.linalg.norm(v)

class WallModel:
    angular_map = {'calovi': lambda angle, p1, p2: np.sin(angle) * (1 + p1 * np.cos(2*angle) + p2*np.cos(4*angle) ),
                   'sin': lambda angle, p1, p2, p3: p1 * np.sin(angle) + p2 * np.sin( 2 * angle ) + p3 * np.sin( 3* angle),
                   'sin-cos': lambda angle, a1, a2, b1, b2 : (a1 * np.sin(angle) + a2 * np.sin(2*angle)) * \
                   (b1 * np.cos(angle) + b2 * np.cos(2*angle) + 1),
                   'shifted-sin-cos': lambda angle, a1, a2, b1, b2:
                   (a1 * np.cos(angle) + a2 * np.sin(2 * (angle + np.pi/2))) *
                   (-b1 * np.sin(angle) + b2 * np.cos(2 * (angle + np.pi/2))+ 1)}
    params_map = {'calovi': np.array([6, 1.0, 0.7, 0.0]),
                  'sin':  np.array([6, 1.0, 0.7, 0.0, 0.0]),
                  'sin-cos': np.array([6, 1.0, 1.0, 1.0, 1.0, 1.0]),
                  'shifted-sin-cos': np.array([6, 1.0, 1.0, 1.0, 1.0, 1.0])}
    
    def __init__(self, angular_model='sin-cos'):
        self.angular_model = angular_model
        self.params = self.params_map[self.angular_model]

        # Unfitted versions
        self._wall_force = lambda dist, decay: np.exp(-(dist/decay)**2 )
        self._wall_repulsion = self.angular_map[self.angular_model]

        # Initialize fitted versions.
        self.set_params(self.params)
    
    def evaluate_raw(self, xdata, *params):
        self.set_params(np.array(params))
        return self.__call__(xdata[0:4], xdata[4:])
    
    def set_params(self, params):
        self.params = params
        self.wall_force = lambda dist: self._wall_force(dist, params[1])
        self.wall_repulsion = lambda angle: self._wall_repulsion(angle, *params[2:])
        self.scale = params[0]
        
    def __call__(self, radius, angle):
        # Sum over all four walls.
        return np.sum(self.scale * self.wall_force(radius) * self.wall_repulsion(angle), axis=0)

def even_fun(angle, *weights):
    result = 1.0
    for i, weight in enumerate(weights):
        result += weight * np.cos((i+1) * angle)
    return result

def odd_fun(angle, *weights):
    result = 0.0
    for i, weight in enumerate(weights):
        result += weight * np.sin((i+1) * angle)
    return result

class SocialModel:
    def __init__(self, num_fourier=2):
        self.num_fourier = num_fourier
        # TODO: Find better initial values.
        params = np.hstack((np.array([0.3, 2.0,1.0]), np.array([1.0] * 3), \
                                 np.zeros(num_fourier * 4)+1))
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
        offset = 5 + num_fun * self.num_fourier
        return self.params[offset:offset+self.num_fourier]
 
class Calovi:
    def __init__(self, wall_model, social_model):
        # TODO: Use correct bbox, for now this is good enough probably.
        self.bounding_box = (0, 30, 0, 30)

        self.position = np.array([15.0, 15.0])
        self.heading = 0 # radians
        self.time = 0
        self.time_after_kick = 0.0
        self.end_position = self.position

        self.wall_distance, self.wall_angle = 0, 0
        self.neigh_distance, self.neigh_viewing_angle, self.neigh_relative_angle = 0, 0, 0

        self.wall_model = wall_model
        self.social_model = social_model

    def get_heading_change(self):
        heading_strength = 0.2 # radians, TODO: Find parameter, use formula 6 for it!
        gaussian = np.random.normal(loc=0.0, scale=1.0)

        wall_heading = self.wall_model(self.wall_distance, self.wall_angle)
        social_heading = self.social_model(self.neigh_distance, self.neigh_viewing_angle, self.neigh_relative_angle)
        
        alpha = 2.0/3.0 # Controls random movement strength near wall, value from Calovi not our data!
        wall_force = np.min(self.wall_model.wall_force(self.wall_distance)) # Weakest wall influence
        random_heading = heading_strength * ( 1- alpha * wall_force) * gaussian
        
        heading_change = self.heading + random_heading + wall_heading + social_heading

        return clip_angle(heading_change)

    def kick_model(self):
        peak_speed = 5 # cm/s
        kick_duration = 0.5 # s
        velocity_decay_time = 0.84179403 # estimated from our data.
        return peak_speed, kick_duration, velocity_decay_time

    def is_inside_arena(self, position):
        x_min, x_max, y_min, y_max = self.bounding_box
        return position[0] > x_min and position[0] < x_max and \
            position[1] > y_min and position[1] < y_max
    
    def kick(self):
        peak_speed, self.kick_duration, self.velocity_decay_time = self.kick_model()
        self.kick_length = peak_speed * self.velocity_decay_time * \
                      (1 - np.exp(-self.kick_duration/self.velocity_decay_time))
        self.position_before_kick = self.position
        self.time_before_kick = self.time
        self.time_after_kick = self.time_before_kick + self.kick_duration

        MAX_TRIALS = 1000
        is_valid_kick = False
        trials = 0
        while not is_valid_kick:
            trials += 1
            # If our model gets stuck, we need a large, random angle change.
            if trials > MAX_TRIALS:
                print("MAX TRIALS reached! Changing to uniform heading!")
                heading_change = np.random.uniform(low=-np.pi, high=np.pi)
            else:
                heading_change = self.get_heading_change()
            self.heading = add_angles(self.heading, heading_change)
            self.end_position = self.position + self.kick_length * angled_vector(self.heading) 
            is_valid_kick = self.is_inside_arena(self.end_position)
        
    def step(self, time):
        assert(time > self.time)
        if self.time_after_kick <= time:
            # First advance to end of current kick.
            self.position = self.end_position
            time = self.time_after_kick
            self.time = time
            # Then calculate new kick.
            self.kick()
            # Then move in direction of the new kick.

        time_diff = time - self.time_before_kick
        # We interpolate between the old and new position.
        drag_coeff = (1 - np.exp(-time_diff/self.velocity_decay_time)) / \
                    (1 - np.exp(-self.kick_duration/self.velocity_decay_time))
        new_position = self.position_before_kick + \
                       self.kick_length * drag_coeff * angled_vector(self.heading) 
        self.position = new_position
        self.time = time
        return time, self.heading, self.position

def main():
    with open('calovi_wall.model', 'rb') as f:
        wall_model = pickle.load(f)
    with open('calovi_social.model', 'rb') as f:
        social_model = pickle.load(f)
    model = Calovi(wall_model=wall_model, social_model=social_model)
    
    for i in range(1,50):
        step = model.step(0.25*i)
        #model.heading = model.get_new_heading()
        print(step)
        print(np.rad2deg(model.heading))

if __name__ == '__main__':
    main()
