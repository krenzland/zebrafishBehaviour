#!/usr/bin/env python3
import dill as pickle
import numpy as np

def unit_vector(v):
    '''Returns a unit vector.'''
    return v/np.linalg.norm(v)

def vector_in_direction(angle):
        x_axis = np.array([-1.0, 0.0]) # right hand coordinate system
        rotation_matrix = np.array([ [np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)] ]) 
        return rotation_matrix @ x_axis

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
        self.wall_force_ = lambda dist, decay: np.exp(-(dist/decay)**2 )
        self.wall_repulsion_ = self.angular_map[self.angular_model]
    
    def evaluate_raw(self, xdata, *params):
        self.set_params(np.array(params))
        return self.__call__(xdata[0:4], xdata[4:])
    
    def set_params(self, params):
        self.params = params
        self.wall_force = lambda dist: self.wall_force_(dist, params[1])
        self.wall_repulsion = lambda angle: self.wall_repulsion_(angle, *params[2:])
        self.scale = params[0]
        
    def __call__(self, radius, angle):
        # Sum over all four walls.
        return np.sum(self.scale * self.wall_force(radius) * self.wall_repulsion(angle), axis=0)

class SocialModel:
    def __init__(self):
        self.social_model = lambda distance, viewing_angle, relative_angle: 0.0

    def __call__(self, distance, viewing_angle, relative_angle):
        return self.social_model(distance, viewing_angle, relative_angle)

class Calovi:
    def __init__(self, wall_model, social_model=None):
        self.position = np.array([15.0, 15.0])
        self.heading = 0 # radians
        self.time = 0

        self.wall_distance, self.wall_angle = 0, 0
        self.neigh_distance, self.neigh_viewing_angle, self.neigh_relative_angle = 0, 0, 0

        if social_model is None:
            social_model = lambda distance, viewing_angle, relative_angle: 0.0
        self.wall_model = wall_model
        self.social_model = social_model

    def get_new_heading(self):
        heading_strength = 0.0 # radians, TODO: Find parameter, use formula 6 for it!
        random_heading = np.random.normal(loc=0.0, scale=1.0)

        wall_heading = self.wall_model(distance, angle)
        social_heading = self.social_model(self.neigh_distance, self.neigh_viewing_angle, self.neigh_relative_angle)
        
        alpha = 2.0/3.0 # Controls random movement strength near wall, value from Calovi not our data!
        wall_force = np.min(self.wall_model.wall_force(self.wall_distance))
        new_heading = self.heading + heading_strength * (1 - wall_force) * random_heading + wall_heading + social_heading
        if new_heading > np.pi:
            new_heading -= 2 * np.pi
        if new_heading < np.pi:
            new_heading += 2 * np.pi

        return new_heading

    def step(self):
        peak_speed = 5 # cm/s
        kick_length = 0.5
        velocity_decay_time = 0.8 #s
        # We could theoretically capture kick_time from data.
        # The three preceeding variables are sufficient to capture the entire motion though.
        # Doing it this way follows the description of Calovi's paper.

        # TODO: Double check formula
        kick_duration = -1 * (np.log((peak_speed * velocity_decay_time - kick_length)/(peak_speed * velocity_decay_time)) / (velocity_decay_time))

        self.time += kick_duration
        self.heading = self.get_new_heading()

        self.position += kick_length * vector_in_direction(self.heading)
        return self.position


def main():
    with open('calovi_wall.model', 'rb') as f:
        wall_model = pickle.load(f)
    social_model = SocialModel()
    model = Calovi(wall_model=wall_model, social_model=social_model)
    
    for i in range(0,100):
        step = model.step()
        #model.heading = model.get_new_heading()
        print(step)
        print(np.rad2deg(model.heading))

if __name__ == '__main__':
    main()
