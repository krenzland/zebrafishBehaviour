#!/usr/bin/env python3
import numpy as np
import pandas as pd
from itertools import product
import scipy.stats
import scipy.signal as signal
import scipy.interpolate as interp
from numba import jit

from util import unit_vector, angle_between, angled_vector, sub_angles

# ----- Util ------

# ----- Loading ------
def load_and_join(csv_path0, csv_path1):
    df_fish0 = pd.read_csv(csv_path0)
    df_fish1 = pd.read_csv(csv_path1)
    cols = ['frame', 'acceleration', 'angle', 'angular_a', 'angular_v',
            'border_distance', 'neighbor_distance', 'speed', 'speed_smooth', 'vX',
            'vY', 'x', 'y', 'time']
    df_fish0.columns = cols
    df_fish1.columns = cols
    df_total = df_fish0.set_index('frame').join(df_fish1, lsuffix='_f0', rsuffix='_f1')
    return df_total

def clean_dataset(df, drop_inf=True):
   # Drop duplicate columns
    df.drop(['time_f1', 'neighbor_distance_f1'], axis=1, inplace=True)
    df.rename(columns={'time_f0': 'time', 'neighbor_distance_f0': 'neighbor_distance'}, inplace=True)
    
    df['neighbor_distance'] = 0.0 # TODO: Remove this column entirely.
    
    # Also drop frames with 0 time difference. Otherwise this destroys the data smoothing!
    dt = np.hstack( (np.ediff1d(df['time']), float('inf')) )
    df = df[dt != 0.0]
    if drop_inf:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        # TODO: If the frame index would be correct, this would create wrong skips.
        # Maybe need to insert interpolated data for dropped frames.
    df.index = range(0, len(df['time']))
    return df

# ----- Smoothing and resampling -----
# Sometimes we have a dataset that is not evenly spaced. this is adjusted by this function here.
def resample_col(data, oldTime, newTime):
    interpolator = interp.CubicSpline(oldTime, data)
    return pd.Series(interpolator(newTime), index=np.arange(0, len(newTime)))

def resample_dataset(df):
    oldTime = df['time'].values
    df['time'] = pd.Series(oldTime, index=np.arange(0, len(oldTime)))
    
    newTime = np.linspace(np.min(oldTime), np.max(oldTime), len(oldTime))
    # Resample each row.
    df = df.apply((lambda col: resample_col(col, oldTime, newTime)), axis=0)
    # Adjust time again
    df['time'] = pd.Series(newTime, index=np.arange(0, len(newTime)))
    return df

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

# In [9]: acc, time
# Out[9]: (array([ 1,  1, -1, -1]), array([0, 1, 2, 3]))

# In [10]: segmentation(acc, time)
# Out[10]: [(0, 0, 0, 1), (0, 2, 2, 1), (2, 3, 1, -1)]

def segmentation(acc, time):
    THRESHOLD = 0.08 #s; treshold for kick duration
    phases = (acc > 0.0) * ACCELERATING
    events = []
    
    idx_start, idx_end, duration, etype = (0, -1, -1, -1)
    cur_sign = -1 * np.sign(acc[0]) # start with correct event
        
    for i, a in enumerate(acc):
        if np.sign(a) != cur_sign or i == (len(acc) -1):        
            # push old event
            idx_end = i
            duration = time[i] - time[idx_start]
            # Ignore short phases, they are probably just some noise.
            if duration < THRESHOLD: # and cur_sign == GLIDING:
                etype = -1 * cur_sign
            else:
                etype = cur_sign
            events.append((idx_start, idx_end, duration, etype))
            
            # start new event
            cur_sign *= -1
            idx_start = i                

    clean_events = []
    # - Remove short phases of same time (from the removal of short phases)
    for i in range(0, len(events)): # skip first invalid element start = end = 0   
        idx_start, idx_end, duration, etype = events[i]
        if i > 0:
            old_idx_start, _, old_duration, old_etype = clean_events[-1]
            if etype == old_etype:
                # Fuse both events
                clean_events[-1] = (old_idx_start, idx_end, old_duration + duration, etype)
                continue
                
        clean_events.append(events[i])

    # - Remove phases that are too short (i.e. < THRESHOLD)
    #clean_events = [ e for e in clean_events if e[2] > THRESHOLD] 
    # TODO: REmove, we do it in summarise kicks currently

    # Skip first, possibly invalid event!
    return clean_events[1:]

def summarise_kick(phase, pos, border_distance):
    start = phase[0][0]
    end = phase[1][1]
    duration = phase[0][2] + phase[1][2]

    if phase[0][2] < 0.08:
        return None
    
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
        kick = summarise_kick(phase, pos, border_distance)
        if kick is not None:
            kicks.append(kick)
        
    # Convert back to frame information (needed for angle calc!)    
    # Some headings are zero. This'll lead to problems later!
    headings = np.zeros( (len(acc), 2) )
    for k in kicks:
        start, end, _, heading, _, _ = k
        headings[start:end] = heading
        
    return kicks, headings

# ---- Angles -----

# TODO: Remove possible dead code here.
@jit
def get_wall_influence_circle(orientation, point, center, radius):
    clostest_point = center + radius * unit_vector(point - center)
    distance = np.linalg.norm(clostest_point - point)
    wall_angle = angle_between(np.array([1,0]), clostest_point - point)
    # Orientation is calculated w.r.t. to [1, 0] from tracking, possible bug.
    relative_angle = sub_angles(wall_angle, orientation)
    return clostest_point, distance, relative_angle

def get_wall_influence(orientation, point, bounding_box):
    # Bounding box here is xMin, xMax, yMin, yMax
    xMin, xMax, yMin, yMax = bounding_box
    # We have 4 walls, wall_0/2 on the x-axis and wall_1/3 on the y-axis.
    wall_axes = np.array( [ 0, 1, 0, 1 ] )
    distance_offset = np.array( [xMin, yMin, xMax, yMax])
    wall_angles = np.deg2rad(np.array( [0, 90, 180, -90] ))

    def dist_to_wall(point, wall_id):
        axis = wall_axes[wall_id]
        return np.abs(distance_offset[wall_id] - [point[axis]] )

    distances = np.array([ dist_to_wall(point, i) for i in range(4) ]).reshape(-1)
    # Orientation is calculated w.r.t. to [1, 0] from tracking, possible bug.
    relative_angles = sub_angles(wall_angles, orientation)
    return distances, relative_angles
   
def calc_angles(kick, pos_0, pos_1, angles_0, angles_1, vel_0, bounding_box, fish_mapping, verbose=False):
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

    # Calculate relevant angles. Note that the viewing angle is NOT symmetric.
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
    
    # Estimate wall information.
    wall_distance_0, wall_angle_0 = get_wall_influence(angles_0[start], pos_f0, bounding_box)
    wall_distance_1, wall_angle_1 = get_wall_influence(angles_1[start], pos_f1, bounding_box)

    kick_information = np.array( [ fish_mapping[0], heading_change, duration, kick_len,  kick_max_vel] )
    social_information = np.array([ dist_norm, dist_angle, geometric_leader, viewing_angle_leader, viewing_angle_follower, rel_orientation ])
    wall_information = np.concatenate( (wall_distance_0, wall_angle_0, wall_distance_1, wall_angle_1) )

    return np.concatenate((kick_information, social_information, wall_information))

# ---- Putting it all together -----
    
def calc_angles_df(df, bounding_box):
    
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
        angles.append(calc_angles(kick, pos0, pos1, df['angle_f0'], df['angle_f1'], df['speed_smooth_f0'], bounding_box, fish_mapping=('f0', 'f1'), verbose=False))
    
    for kick in kicks1:
        angles.append(calc_angles(kick, pos1, pos0, df['angle_f1'], df['angle_f0'], df['speed_smooth_f1'], bounding_box, fish_mapping=('f1', 'f0'), verbose=False))

    kick_columns = [ 'fish_id', 'heading_change', 'duration', 'length', 'max_vel']
    social_columns = ['neighbor_distance', 'neighbor_angle', 'geometric_leader', 'viewing_angle_ltf', 'viewing_angle_ftl', 'rel_orientation']
    wall_columns = [ f"wall_{type}{wall}_{id}" for id, type, wall in product( ['f0', 'f1'], ['distance', 'angle'],[0,1,2,3] )]
    columns = kick_columns + social_columns + wall_columns
    df_kicks = pd.DataFrame(data=angles, columns=columns)
    
    return df_kicks

def main():
    # Constants
    BODY_LENGTH = 1.0 # cm
    SWIMMING_THRESHOLD = 0.5/BODY_LENGTH
    WALL_CENTER = np.array([15,15])
    WALL_RADIUS = 14
    csv_path_0 = '../data/raw/trial3_fish0.csv'
    csv_path_1 = '../data/raw/trial3_fish1.csv'
    csv_cleaned = '../data/processed/cleaned_guy.csv'
    csv_kicks = '../data/processed/kicks_guy.csv'

    df = load_and_join(csv_path_0, csv_path_1)
    print("Loading and joining done.")
    df = clean_dataset(df)
    print("Cleaned data.")
    #print(df.describe())
    df = resample_dataset(df)
    df = smooth_dataset(df)
    print("Smoothed velocity and acceleration!")
    df = add_status(df, SWIMMING_THRESHOLD)
    print("Found active and passive swimming phases.")
    print(f"The data-frame has the following columns now:{df.columns}")


    # Calculate bounding box for rectangular arena.
    bounding_box = (min(df['x_f0'].min(), df['x_f1'].min()), max(df['x_f0'].max(), df['x_f1'].max()), 
                    min(df['y_f0'].min(), df['y_f1'].min()), max(df['y_f0'].max(), df['y_f1'].max()))
    
    
    df_kicks = calc_angles_df(df, bounding_box)

    print("Calculated angles.")
    print(f"The kicks_df has the following columns now:{df_kicks.columns}")
    df.to_csv(csv_cleaned, index=False)
    df_kicks.to_csv(csv_kicks, index=False)

if __name__ == '__main__':
    main()
