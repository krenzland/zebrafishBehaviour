#!/usr/bin/env python3
import numpy as np
import pandas as pd
import scipy.stats
import scipy.signal as signal
from numba import jit

from util import unit_vector, angle_between, angled_vector, sub_angles

# ----- Util ------

# ----- Loading ------
def load_and_join(csv_path0, csv_path1):
    df_fish0 = pd.read_csv(csv_path0)
    df_fish1 = pd.read_csv(csv_path1)
    cols = ['frame', 'acceleration', 'angle', 'aX', 'aY',
            'border_distance', 'neighbor_distance',
            'speed', 'vX', 'vY', 'x', 'y',
            'time']
    df_fish0.columns = cols
    df_fish1.columns = cols
    df_total = df_fish0.set_index('frame').join(df_fish1, lsuffix='_f0', rsuffix='_f1')
    return df_total

def clean_dataset(df, drop_inf=True):
   # Drop duplicate columns
    df.drop(['time_f1', 'neighbor_distance_f1'], axis=1, inplace=True)
    df.rename(columns={'time_f0': 'time', 'neighbor_distance_f0': 'neighbor_distance'}, inplace=True)
    if drop_inf:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # TODO: If the frame index would be correct, this would create wrong skips.
        # Maybe need to insert interpolated data for dropped frames.
        df.index = range(0, len(df['time']))
    return df

# ----- Smoothing -----
def smooth_vector(x):    
    # TODO: Double check whether window_length correspond to a reasonable timeframe!
    degree = 3
    window_length=49
    
    x = signal.savgol_filter(x, window_length=window_length, polyorder=degree, deriv=0) # TODO: Check
    x_deriv = signal.savgol_filter(x, window_length=window_length, polyorder=degree, deriv=1)
    
    return x, x_deriv

def smooth_dataset(df):
    vel0 = df['speed_f0'].values
    vel1 = df['speed_f1'].values
    vel_smooth0, acc_smooth0 = smooth_vector(vel0)
    vel_smooth1, acc_smooth1 = smooth_vector(vel1)

    df.loc[:,'speed_smooth_f0'] = pd.Series(vel_smooth0, index=df.index)
    df.loc[:,'speed_smooth_f1'] = pd.Series(vel_smooth1, index=df.index)
    df.loc[:,'acceleration_smooth_f0'] = pd.Series(acc_smooth0, index=df.index)
    df.loc[:,'acceleration_smooth_f1'] = pd.Series(acc_smooth1, index=df.index)

    return df

# ---- Activity selection -----

@jit(nopython=False)
def get_status(vel0, vel1, time, treshold):
    # The constants are used internally to speed up computation.
    SWIMMING = 0
    PAUSING = 1
    STOPPING = 2

    time_window = 4.0
    vel_max = np.maximum(vel0, vel1)
    
    time_paused = np.zeros(vel_max.shape)
    
    # Forward pass: Find out how long fish have been pausing
    cur_time_paused = 0
    for i in range(0, len(vel_max)):
        if vel_max[i] >= treshold:
            # Swimming
            cur_time_paused = 0
        else:
            # Pausing or stopping
            if i > 0:
                cur_time_paused += time[i] - time[i - 1]
            else:
                cur_time_paused += time[i]
            #print(cur_time_paused, time[i])
        time_paused[i] = cur_time_paused
        
    # Backwards pass: Set status flags
    status = np.zeros(vel_max.shape)
    is_cur_stopped = False
    for i in range(len(vel_max) - 1, -1, -1):
            if time_paused[i] == 0:
                # Swimming
                is_cur_stopped = False
                status[i] = SWIMMING
            elif is_cur_stopped:
                # Still stopping
                status[i] = STOPPING
            elif time_paused[i] >= treshold:
                # Paused for too long -> Stopping
                is_cur_stopped = True
                status[i] = STOPPING
            else:
                status[i] = PAUSING                    

    pretty_status = np.zeros(status.shape, dtype=np.object)
    pretty_status[np.where(status == SWIMMING)] = 'swimming'
    pretty_status[np.where(status == PAUSING)] = 'pausing'
    pretty_status[np.where(status == STOPPING)] = 'stopping'
                    
    return time_paused, pretty_status

def add_status(df, treshold):
    vel0 = df['speed_smooth_f0'].values
    vel1 = df['speed_smooth_f1'].values
    time = df['time'].values
    _time_paused, status = get_status(vel0, vel1, time, treshold)
    df.loc[:, 'status'] = pd.Series(status, index=df.index)
    return df

# ----- Segmentation ----
ACCELERATING = 1
GLIDING = -1

@jit(nopython=True)
def segmentation(acc, time):
    phases = (acc > 0.0) * ACCELERATING
    # TODO (maybe): fuse nearby accelerating events
    
    events = []
    
    # (idx_start, idx_end, duration, type)    
    idx_start, idx_end, duration, etype = (0, -1, -1, -1)
    cur_sign = -1 * np.sign(acc[0]) # start with correct event
        
    for i, a in enumerate(acc):
        if np.sign(a) != cur_sign or i == (len(acc) -1):        
            # push old event
            idx_end = i - 1
            duration = time[i - 1] - time[idx_start]
            events.append((idx_start, idx_end, duration, cur_sign))
            
            # start new event
            cur_sign *= -1
            idx_start = i                
        
    return events[1:] # skip first invalid element start = end = 0   

def summarise_kick(phase, pos, border_distance):
    start = phase[0][0]
    end = phase[1][1]
    duration = phase[0][2] + phase[1][2]
    
    pos_start = np.array([ pos[0][start], pos[1][start] ])
    pos_end = np.array([ pos[0][end], pos[1][end] ])

    traj_kick = pos_end - pos_start
    heading = unit_vector(traj_kick)
    kick_len = np.linalg.norm(traj_kick)
   
    return start, end, duration, heading, kick_len, border_distance

def summarise_kicks(pos, acc, border_distance, time):
    phases = segmentation(acc, time)
    # If heading is zero (no current kick) need to estimate traj. differently!
    
    # Find first acceleration phase
    begin = 0
    for i in range(0, len(phases)):
        if phases[i][-1] == 1:
            begin = i
            break
    print(f"Begin={begin}")    
    
    kicks = []
    for i in range(begin, len(phases)//2):
        # A phase consists in a acceleration + gliding phase
        phase = phases[i*2 : i*2+2]
        kicks.append(summarise_kick(phase, pos, border_distance))
        
    # Convert back to frame information (needed for angle calc!)    
    # Some headings are zero. This'll lead to problems later!
    headings = np.zeros( (len(acc), 2) )
    for k in kicks:
        start, end, _, heading, _, _ = k
        headings[start:end] = heading
        
    return kicks, headings

# ---- Angles -----

@jit
def get_wall_influence(orientation, point, center, radius):
    clostest_point = center + radius * unit_vector(point - center)
    distance = np.linalg.norm(clostest_point - point)
    wall_angle = angle_between(np.array([-1,0]), clostest_point - point)
    # Orientation is calculated w.r.t. to [1, 0] from tracking, possible bug.
    relative_angle = sub_angles(wall_angle, -orientation)
    return clostest_point, distance, relative_angle

def calc_angles(kick, pos_0, pos_1, angles_0, angles_1, vel_0, wall_fun, fish_mapping, verbose=False):
    x_axis = np.array([1, 0]) # Used as a common reference for angles.
    
    start, end, duration, heading, kick_len, _ = kick
    
    pos_f0 = np.array([ pos_0[0][start], pos_0[1][start] ])
    pos_f1 = np.array([ pos_1[0][start], pos_1[1][start] ])
 
    # Traj. vector of both fish
    # We use the angle of the fish, as it is more stable than the velocity.
    traj_f0 = angled_vector(angles_0[start])
    traj_f1 = angled_vector(angles_1[start])
  
    # Vector connecting both fish, note that it is directed!
    dist = pos_f1 - pos_f0
    dist_norm = np.linalg.norm(dist)
    dist_angle = angle_between(x_axis, dist)

    # Calculate relevant angles and focal fish
    viewing_angle_0t1 = sub_angles(dist_angle, angles_0[start])
    viewing_angle_1t0 = sub_angles(-dist_angle, angles_1[start])
    
    # The focal fish is defined as the geometric leader, i.e. the fish with the larger viewing angle.
    # Corresponds to the fish which would need to turn more to look directly at the other fish.
    # The sign of the relative orientation depends on which fish is the focal one.
    if np.abs(viewing_angle_0t1) > np.abs(viewing_angle_1t0):
        geometric_leader = fish_mapping[0]
        rel_orientation = sub_angles(angles_1[start], angles_0[start])
        viewing_angle_leader, viewing_angle_follower = viewing_angle_0t1, viewing_angle_1t0
    else:
        geometric_leader = fish_mapping[1]
        rel_orientation = sub_angles(angles_0[start], angles_1[start])
        viewing_angle_leader, viewing_angle_follower = viewing_angle_1t0, viewing_angle_0t1
                         
    # Collect some other summary data about the kick
    traj_kick = np.array([ pos_0[0][end], pos_0[1][end] ]) - pos_f0
    kick_len = np.linalg.norm(traj_kick)
    kick_heading = angle_between(x_axis, traj_kick)
    kick_max_vel = np.max(vel_0[start:end])
    heading_change = sub_angles(angles_0[end], angles_0[start])
    
    # Wall distance and angle
    #def get_wall_influence(orientation, point, center, radius):
    clostest_point_0, wall_distance_0, wall_angle_0 = wall_fun(angles_0[start], pos_f0)
    clostest_point_1, wall_distance_1, wall_angle_1 = wall_fun(angles_1[start], pos_f1)

    if verbose:
        print(f"Max swimming velocity {kick_max_vel}")
        print("Start = {}, end {}, duration = {}".format(start, end, duration))
        print("Trajectories: f0 = {}, f1 = {}".format(unit_vector(traj_f0), unit_vector(traj_f1)))
        print("x/y: f0 = {}, f1 = {}".format(pos_f0, pos_f1))
        print("Distance = {}, norm = {}, data = {}".format(dist, np.linalg.norm(dist), df_total['neighbor_distance'][start]))

        print("Viewing angles: 0->1 = {:3.2f}°, 1->0 = {:3.2f}°".format(
            np.rad2deg(viewing_angle_0t1), np.rad2deg(viewing_angle_1t0)))
        print("Geometric leader is {}.".format(geometric_leader))
        print("Relative orientation = {:3.2f}°".format(np.rad2deg(rel_orientation)))
        print("Heading = {}, kick_len = {}".format(unit_vector(traj_kick), np.linalg.norm(traj_kick)))
        
        print(f"Wall distance: 0 = {wall_distance_0}, 1 = {wall_distance_1}")
        print(f"Wall angle: 0 = {np.rad2deg(wall_angle_0)}, 1 = {np.rad2deg(wall_angle_1)}")
   

    kick_information = np.array( [ fish_mapping[0], heading_change, duration, kick_len,  kick_max_vel] )
    social_information = np.array([ dist_norm, dist_angle, geometric_leader, viewing_angle_leader, viewing_angle_follower, rel_orientation ])
    wall_information = np.array( [ wall_distance_0, wall_angle_0, wall_distance_1, wall_angle_1 ])

    return np.concatenate((kick_information, social_information, wall_information))

# ---- Putting it all together -----
    
def calc_angles_df(df, wall_fun):
    pos0 = (df['x_f0'], df['y_f0'])
    pos1 = (df['x_f1'], df['y_f1'])
    border_distance0 =  df['border_distance_f0'].values
    border_distance1 =  df['border_distance_f1'].values
    acc_smooth0 = df['acceleration_smooth_f0'].values
    acc_smooth1 = df['acceleration_smooth_f1'].values
    time = df['time'].values
    kicks0, _ = summarise_kicks(pos0, acc_smooth0, border_distance0, time)
    kicks1, _ = summarise_kicks(pos1, acc_smooth1, border_distance1, time)
    print("Summarised kicks.")
    
    angles = []
    for kick in kicks0:
        angles.append(calc_angles(kick, pos0, pos1, df['angle_f0'], df['angle_f1'], df['speed_smooth_f0'], wall_fun, fish_mapping=('f0', 'f1'), verbose=False))
    
    for kick in kicks1:
        angles.append(calc_angles(kick, pos1, pos0, df['angle_f1'], df['angle_f0'], df['speed_smooth_f1'], wall_fun, fish_mapping=('f1', 'f0'), verbose=False))

    kick_columns = [ 'fish_id', 'heading_change', 'duration', 'length', 'max_vel']
    social_columns = ['neighbor_distance', 'neighbor_angle', 'geomentric_leader', 'viewing_angle_ltf', 'viewing_angle_ftl', 'rel_orientation']
    wall_columns = [ 'wall_distance_f0', 'wall_angle_f0', 'wall_distance_f1', 'wall_angle_f1']
    columns = kick_columns + social_columns + wall_columns
    df_kicks = pd.DataFrame(data=angles, columns=columns)
    
    return df_kicks

def main():
    # Constants
    BODY_LENGTH = 0.64 # cm
    SWIMMING_THRESHOLD = 0.5/BODY_LENGTH
    WALL_CENTER = np.array([15,15])
    WALL_RADIUS = 14
    csv_path_0 = '../data/raw/zebrafish26.01.2017_fish0.csv'
    csv_path_1 = '../data/raw/zebrafish26.01.2017_fish1.csv'
    csv_kicks = '../data/processed/kicks26.01.2017.csv'

    
    df = load_and_join(csv_path_0, csv_path_1)
    print("Loading and joining done.")
    df = clean_dataset(df)
    print("Cleaned data.")
    #print(df.describe())
    df = smooth_dataset(df)
    print("Smoothed velocity and acceleration!")
    df = add_status(df, SWIMMING_THRESHOLD)
    print("Found active and passive swimming phases.")
    print(f"The data-frame has the following columns now:{df.columns}")

    wall_fun = lambda orientation, point: get_wall_influence(orientation, point, WALL_CENTER, WALL_RADIUS)
    df_kicks = calc_angles_df(df, wall_fun)

    print("Calculated angles.")
    print(f"The kicks_df has the following columns now:{df_kicks.columns}")
    df_kicks.to_csv(csv_kicks, index=False)

if __name__ == '__main__':
    main()
