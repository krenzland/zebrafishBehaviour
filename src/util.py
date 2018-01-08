import numpy as np
from numba import jit


# angle_between from https://stackoverflow.com/a/13849249
@jit
def unit_vector(vector):
    """ Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)

@jit
def angle_between(v1, v2, signed=True):
    """ Returns the angle in radians between vectors 'v1' and 'v2'. """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    if signed:
        return np.arctan2(np.cross(v1_u, v2_u), np.dot(v1_u, v2_u))
    else:
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
@jit  
def angled_vector(angle, reference=np.array([1, 0])):
    """ Calculate a unit vector of angle 'angle' w.r.t the vector 'reference'. """
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([c, -s, s, -c]).reshape(2,2)
    return (R @ reference.reshape(2,1)).reshape(2)

@jit
def sub_angles(a, b):
    """Subtract two angles whilst clipping them into a reasonable range."""
    return (a - b + np.pi + 2*np.pi) % (2 * np.pi) - np.pi

@jit
def add_angles(a, b):
    # TODO: Maybe refactor to sub_angles(a, -b) or sth.
    angle = a + b
    return clip_angle(angle)

@jit
def clip_angle(angle):
    while angle < -np.pi:
        angle += 2*np.pi
    while angle > np.pi:
        angle -= 2*np.pi
    return angle
