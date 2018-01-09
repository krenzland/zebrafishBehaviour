#!/usr/bin/env python3
import dill as pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from util import add_angles, angled_vector, clip_angle, unit_vector
from segmentation import get_wall_influence

class KickModel:
    def __init__(self, peak_speed_model, kick_duration_model, velocity_decay_time):
        self.peak_speed_model = peak_speed_model
        self.kick_duration_model = kick_duration_model
        self.velocity_decay_time = velocity_decay_time
    
    def get_peak_speed(self):
        return self.peak_speed_model.sample()[0][0,0]
    
    def get_kick_duration(self):
        return self.kick_duration_model.sample()[0][0,0]
    
    def get_velocity_decay_time(self):
        return self.velocity_decay_time

class WallModel:
    angular_map = {'calovi': lambda angle, p1, p2: np.sin(angle) * (1 + p1 * np.cos(2*angle) + p2*np.cos(4*angle) ),
                   'sin': lambda angle, p1, p2, p3: p1 * np.sin(angle) + p2 * np.sin( 2 * angle ) + p3 * np.sin( 3* angle),
                   'sin-cos': lambda angle, a1, a2, b1, b2 : (a1 * np.sin(angle) + a2 * np.sin(2*angle)) * \
                   (b1 * np.cos(angle) + b2 * np.cos(2*angle) + 1),
                   'shifted-sin-cos': lambda angle, a1, a2, b1, b2:
                   (a1 * np.cos(angle) + a2 * np.sin(2 * (angle + np.pi/2))) *
                   (-b1 * np.sin(angle) + b2 * np.cos(2 * (angle + np.pi/2))+ 1)}
    params_map = {'calovi': np.array([1, .0, 0.7, 0.0]),
                  'sin':  np.array([1, 6.0, 0.7, 0.0, 0.0]),
                  'sin-cos': np.array([1, 6.0, 1.0, 1.0, 1.0, 1.0]),
                  'shifted-sin-cos': np.array([1, 6.0, 1.0, 1.0, 1.0, 1.0])}
    
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
    def __init__(self, kick_model, wall_model, social_model):
        # TODO: Use correct bbox, for now this is good enough probably.
        self.bounding_box = (0, 30, 0, 30)

        self.position = np.array([5.0, 5.0])
        self.heading = 0 # radians
        self.time = 0
        self.time_after_kick = 0.0
        self.end_position = self.position

        self.wall_distances, self.wall_angles = 0, 0
        self.neigh_distance, self.neigh_viewing_angle, self.neigh_relative_angle = 0, 0, 0

        self.kick_model = kick_model
        self.wall_model = wall_model
        self.social_model = social_model

    def get_heading_change(self):
        heading_strength = 1.02 # radians, TODO: Find a better estimate, this one is very noisy!
        gaussian = np.random.normal(loc=0.0, scale=1.0)

        wall_heading = self.wall_model(self.wall_distances, self.wall_angles)
        #print(f"Wall heading = {np.rad2deg(wall_heading)}")
        social_heading = self.social_model(self.neigh_distance, self.neigh_viewing_angle, self.neigh_relative_angle)
        
        alpha = 2.0/3.0 # Controls random movement strength near wall, value from Calovi not our data!
        wall_force = np.max(self.wall_model.wall_force(self.wall_distances)) # Strongest wall influence
        random_heading = heading_strength * ( 1- alpha * wall_force) * gaussian
        
        heading_change = self.heading + random_heading + wall_heading + social_heading

        return clip_angle(heading_change)

    def is_inside_arena(self, position):
        x_min, x_max, y_min, y_max = self.bounding_box
        return position[0] > x_min and position[0] < x_max and \
            position[1] > y_min and position[1] < y_max
    
    def update_environment(self):
        self.wall_distances, self.wall_angles = get_wall_influence(self.heading, self.position, \
                                                                    self.bounding_box)
    
    def kick(self):
        # First we need to update all distances angles.
        self.update_environment()
        
        peak_speed = self.kick_model.get_peak_speed()
        self.kick_duration = self.kick_model.get_kick_duration()
        self.velocity_decay_time = self.kick_model.get_velocity_decay_time()

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
    
def add_to_buffer(buffer, value):
    buffer_local = np.roll(buffer, shift=-1)
    buffer_local[-1] = value
    np.copyto(dst=buffer, src=buffer_local)
    return buffer

def animate(model, n_frames, filename):
     # Set up initial values for animation.
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(xlim=(0, 30), ylim=(0, 30))
    lines0, = ax.plot([], [], c='red', label='our fish')
    ax.legend(loc='upper right')

    # Set up animation buffers.
    visible_steps = 10
    past_x0 = np.zeros(visible_steps) + 5
    past_y0 = np.zeros(visible_steps) + 5

    def init():
        lines0.set_data([], [])
        return lines0, 

    def animate(i):
        _, _, (x0, y0) = model.step((i+1) * 0.1)

        add_to_buffer(past_x0, x0)
        add_to_buffer(past_y0, y0)

        lines0.set_data(past_x0, past_y0)

        return lines0,

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=n_frames, interval=100, blit=True)
    anim.save(filename)

def main():
    with open('calovi_kick.model', 'rb') as f:
        kick_model = pickle.load(f)
    with open('calovi_wall.model', 'rb') as f:
        wall_model = pickle.load(f)
    with open('calovi_social.model', 'rb') as f:
        social_model = pickle.load(f)
    model = Calovi(kick_model=kick_model, wall_model=wall_model, social_model=social_model)
    animate(model, 500, 'wall_animation.mp4')

if __name__ == '__main__':
    main()
