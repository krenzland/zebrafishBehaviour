#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from itertools import product
import scipy.stats
import scipy.signal as signal
import scipy.interpolate as interp
from numba import jit
import sklearn.model_selection as cv

from util import unit_vector, angle_between, angled_vector, sub_angles, add_angles

# ----- Util ------

# ----- Loading ------
def load_and_join(csv_path0, csv_path1):
    df_fish0 = pd.read_csv(csv_path0)
    df_fish1 = pd.read_csv(csv_path1)
    # Old cols frame	ACCELERATION#wcentroid (cm/s2)	ANGLE#wcentroid	ANGULAR_A#centroid	ANGULAR_V#centroid	BORDER_DISTANCE#wcentroid (cm)
    #NEIGHBOR_DISTANCE (cm)	SPEED#wcentroid (cm/s)	SPEED#smooth#wcentroid (cm/s)	VX#wcentroid (cm/s)	VY#wcentroid (cm/s)
    # X#wcentroid (cm)	Y#wcentroid (cm)	time#centroid

#     cols = ['frame', 'acceleration', 'acceleration_smooth', 'acceleration_wcentroid', 'angle', 'angular_a', 'angular_v',
#             'aX', 'aY'
#             'border_distance',
#             'midline_offset',
#             'neighbor_distance', 'speed', 'speed_smooth', 'vX',
#             'vY', 'x', 'y', 'time']
    cols = ['frame', 'acceleration', 'acceleration_smooth', 'acceleration_smooth_wcentroid', 'angle', 'angular_a', 'angular_v',
            'aX', 'aY', 'border_distance',
            'midline_offset', 'neighbor_distance', 'SPEED#wcentroid (cm/s)',
            'speed_smooth_wcentroid', 'speed__pcentroid',
            'speed_smooth_pcentroid', 'speed', 'speed_smooth',
            'vX', 'vY', 'x_wcentroid', 'x',
            'y_wcentroid', 'y', 'normalized_midline', 'time']

    drop =['acceleration_smooth', 'acceleration_smooth_wcentroid', 'angular_a', 'angular_v',
            'aX', 'aY', 'border_distance',
            'midline_offset', 'neighbor_distance', 'SPEED#wcentroid (cm/s)',
            'speed_smooth_wcentroid', 'speed__pcentroid',
            'speed_smooth_pcentroid',  'speed_smooth',
            'vX', 'vY', 'x_wcentroid', 
            'y_wcentroid', 'normalized_midline']
    df_fish0.columns = cols
    df_fish1.columns = cols
    df_fish0.drop(drop, axis=1, inplace=True)
    df_fish1.drop(drop, axis=1, inplace=True)
    df_total = df_fish0.set_index('frame').join(df_fish1, lsuffix='_f0', rsuffix='_f1')
    return df_total

def fix_time(df):
    # Chage time so that it is equally spaced.
    # Some frames are marked as invalid, those have to be dropped for kicks.

    THRESHOLD = 0.005 # time differences smaller than this are ignored.
    dt = 0.01 # difference between two consecutive frames
    time = df['time']
    offset = np.ceil(df['time'][0]/dt) * dt

    new_time = []
    valid = []
    i = 0
    while(i < len(time)):
        cur_adj_time = len(new_time) * dt + offset
        new_time.append(cur_adj_time)

        if (time[i] - cur_adj_time) > THRESHOLD:
            # There is some missing frame here!
            valid.append(False)
        else:
            valid.append(True)
            i += 1

    new_time = np.array(new_time)
    valid = np.array(valid)

    # Now we have to include the new time in the dataframe, without destroying data.
    new_index = pd.RangeIndex(0, len(new_time), step=1)
    # Prepare old data for new time
    df.index = new_index[valid]
    df['time'] = new_time[valid]

    # Create frames for invalid entries (all NaN!)
    invalid_frame = pd.DataFrame(index=new_index[~valid])
    # Combine invalid frames and valid ones, recreate sorted index
    df = df.append(invalid_frame).sort_index()
    # Store which frames are valid, this is needed for the kick-segmentation later.
    df['dropped'] = False
    df.loc[~valid, 'dropped'] = True 
    # Store correct time for invalid frames as well
    df['time'] = new_time

    return df

def clean_dataset(df, drop_inf=True):
   # Drop duplicate columns
    df = df.drop(['time_f1'], axis=1)
    df = df.rename(columns={'time_f0': 'time'})
    
    if drop_inf:
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
    # Fix index.
    df.index = range(0, len(df['time']))

    df = fix_time(df)
    return df


# ----- Smoothing and resampling -----
# Sometimes we have datasets with dropped frames.
# We interpolate the values for them to decrease problems with smoothing later.
# This isn't mathematically elegant but not important because those frames aren't
# considered for kicks anyway!
def interpolate_col(data, time, dropped):
    valid_times = time[~dropped]
    invalid_times = time[dropped]
    
    # Compute cubic spline with valid data.
    interpolator = interp.CubicSpline(valid_times, data[~dropped])
    # Replace invalid data with data from interpolation
    index = data.index
    data = data.values
    data[dropped] = interpolator(invalid_times)
    
    return pd.Series(data, index)

def interpolate_invalid(df):
    # Save columns. Otherwise we loose information about dropped frames!
    time = np.copy(df['time'].values)
    dropped = np.copy(df['dropped'].values)
    
    # Resample invalid entries.
    df = df.apply((lambda col: interpolate_col(col, df['time'].values, df['dropped'].values)), axis=0)

    # Restore columns.
    df['time'] = time
    df['dropped'] = dropped
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
            elif time_paused[i] >= time_window:
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

    # Skip first, possibly invalid event!
    return clean_events[1:]

def summarise_kick(phase, pos, status, dropped):
    start = phase[0][0]
    end = phase[1][1]
    duration = phase[0][2] + phase[1][2]
    gliding_duration = phase[1][2]

    # Discard kick if acceleration phase is too short.
    if phase[0][2] < 0.08:
        return None

    pos_start = np.array([ pos[0][start], pos[1][start] ])
    pos_end = np.array([ pos[0][end], pos[1][end] ])

    # Discard kick if it contains stopping.
    if np.any(status[start:end] == 'stopping'):
        #print('Stopping occured!')
        return None

    # Discard kick if it contains dropped frames
    if np.any(dropped[start:end]):
        #print('Kick contains dropped frames!')
        return None

    # Kick is now a valid kick.
    traj_kick = pos_end - pos_start
    heading = unit_vector(traj_kick)
    kick_len = np.linalg.norm(traj_kick)
   
    return start, end, duration, gliding_duration, heading, kick_len

def summarise_kicks(pos, acc, status, dropped, time):
    phases = segmentation(acc, time)
    
    # Find first acceleration phase
    begin = 0
    for i in range(0, len(phases)):
        if phases[i][-1] == 1:
            begin = i
            break
    print(f"Begin={begin}")    
    
    kicks = []
    discarded = 0
    for i in range(begin, len(phases)//2):
        # A phase consists in a acceleration + gliding phase
        phase = phases[i*2 : i*2+2]
        kick = summarise_kick(phase, pos, status, dropped)
        if kick is not None:
            kicks.append(kick)
        else:
            discarded += 1
    print(f'Removed {discarded} kicks.') 
    return kicks

# ---- Angles -----
def get_wall_influence(orientation, point, bounding_box):
    # Bounding box here is xMin, xMax, yMin, yMax
    xMin, xMax, yMin, yMax = bounding_box
    # We have 4 walls, wall_0/2 on the x-axis and wall_1/3 on the y-axis.
    wall_axes = np.array( [ 1, 0, 1, 0 ] )
    distance_offset = np.array( [yMin, xMax, yMax, xMin ])
    wall_angles = np.deg2rad(np.array( [-90, 0, 90, 180] ) + 90)

    def dist_to_wall(point, wall_id):
        axis = wall_axes[wall_id]
        return np.abs(distance_offset[wall_id] - [point[axis]] )

    distances = np.array([ dist_to_wall(point, i) for i in range(4) ]).reshape(-1)
    # Orientation is calculated w.r.t. to [1, 0] by me.
    relative_angles = add_angles(wall_angles, -orientation)
    return distances, relative_angles
   
def calc_angles(kick, pos_0, pos_1, angles_0, angles_1, vel_0, bounding_box, valid, fish_mapping, num_past_window=2, verbose=False):
    x_axis = np.array([1, 0]) # Used as a common reference for angles.
    
    start, end, duration, gliding_duration, heading, kick_len = kick

    # Check if we have enough information about the past.
    if start - num_past_window < 0:
        return None
    
    kick_information = None
    rows = []
    for dt in range(num_past_window+1):
        pos_f0 = np.array([ pos_0[0][start - dt], pos_0[1][start - dt] ])
        pos_f1 = np.array([ pos_1[0][start - dt], pos_1[1][start - dt] ])

        # Kick information:
        # Extract this only for dt = 0.
        # Otherwise reuse same value - not tidy, but makes analysis easier.
        # These are the statistics about the kick itself.
        if dt == 0:
            traj_kick = np.array([ pos_0[0][end], pos_0[1][end] ]) - pos_f0
            kick_len = np.linalg.norm(traj_kick)
            # TODO: Fix potential off by one error here.
            kick_max_vel = np.max(vel_0[start:end+1])
            end_vel = vel_0[end]#np.min(vel_0[start:end])
            heading_change = sub_angles(angles_0[end], angles_0[start])

            kick_information = np.array( [ fish_mapping[0], heading_change, duration, gliding_duration, kick_len, kick_max_vel, end_vel] )
            if not valid[start]:
                print('Kick invalid!')
        
        if not valid[start - dt]:
            return None
            
        # Social information:
        # Vector connecting both fish, note that it is directed!
        dist = pos_f1 - pos_f0
        dist_norm = np.linalg.norm(dist)
        dist_angle = angle_between(x_axis, dist)

        # Calculate relevant angles. Note that the viewing angle is NOT symmetric.
        viewing_angle_0t1 = sub_angles(dist_angle, angles_0[start - dt])
        viewing_angle_1t0 = sub_angles(-dist_angle, angles_1[start - dt])

        # The focal fish is defined as the geometric leader, i.e. the fish with the larger viewing angle.
        # Corresponds to the fish which would need to turn more to look directly at the other fish.
        # The sign of the relative orientation depends on which fish is the focal one.
        if np.abs(viewing_angle_0t1) > np.abs(viewing_angle_1t0):
            geometric_leader = fish_mapping[0]
            rel_orientation = sub_angles(angles_1[start - dt], angles_0[start - dt])
            viewing_angle_leader, viewing_angle_follower = viewing_angle_0t1, viewing_angle_1t0
        else:
            geometric_leader = fish_mapping[1]
            rel_orientation = sub_angles(angles_0[start - dt], angles_1[start - dt])
            viewing_angle_leader, viewing_angle_follower = viewing_angle_1t0, viewing_angle_0t1

        social_information = np.array([ dist_norm, dist_angle, geometric_leader, viewing_angle_leader, viewing_angle_follower, rel_orientation ])

        # Estimate wall information.
        wall_distance_0, wall_angle_0 = get_wall_influence(angles_0[start - dt], pos_f0, bounding_box)
        wall_distance_1, wall_angle_1 = get_wall_influence(angles_1[start - dt], pos_f1, bounding_box)

        wall_information = np.concatenate( (wall_distance_0, wall_angle_0, wall_distance_1, wall_angle_1) )

        row = np.concatenate(([dt], kick_information, social_information, wall_information))
        rows.append(row)

    return np.array(rows)

# ---- Putting it all together -----
def calc_angles_df(df, fish_names, bounding_box):
    pos0 = (df['x_f0'], df['y_f0'])
    pos1 = (df['x_f1'], df['y_f1'])
    acc_smooth0 = df['acceleration_smooth_f0'].values
    acc_smooth1 = df['acceleration_smooth_f1'].values
    status = df['status'].values
    dropped = df['dropped'].values
    time = df['time'].values
    kicks0 = summarise_kicks(pos0, acc_smooth0, status, dropped, time)
    kicks1 = summarise_kicks(pos1, acc_smooth1, status, dropped, time)
    print("Summarised kicks.")
    valid = (df['status'] != 'stopping') & (~df['dropped'])
    
    angles = []
    for kick in kicks0:
        angles.append(calc_angles(kick, pos0, pos1, df['angle_f0'], df['angle_f1'], df['speed_smooth_f0'], bounding_box, valid, fish_mapping=fish_names, verbose=False))
    
    for kick in kicks1:
        angles.append(calc_angles(kick, pos1, pos0, df['angle_f1'], df['angle_f0'], df['speed_smooth_f1'], bounding_box, valid, fish_mapping=fish_names[::-1], verbose=False))

    # Remove invalid kicks
    angles = np.concatenate([a for a in angles if a is not None])
        
    kick_columns = [ 'fish_id', 'heading_change', 'duration', 'gliding_duration', 'length', 'max_vel', 'end_vel']
    social_columns = ['neighbor_distance', 'neighbor_angle', 'geometric_leader', 'viewing_angle_ltf', 'viewing_angle_ftl', 'rel_orientation']
    wall_columns = [ f"wall_{type}{wall}_{id}" for id, type, wall in product( ['f0', 'f1'], ['distance', 'angle'],[0,1,2,3] )]
    columns = ['dt'] + kick_columns + social_columns + wall_columns
    df_kicks = pd.DataFrame(data=angles, columns=columns)
    
    return df_kicks

def main():
    # Constants
    BODY_LENGTH = 1.0 # cm
    SWIMMING_THRESHOLD = 0.5/BODY_LENGTH

    # {} in csv is placeholder for train/test.
    csv_cleaned = '../data/processed/cleaned_guy_{}.csv'
    csv_kicks = '../data/processed/kicks_guy_{}.csv'

    trials = range(2,11+1)
    csv_dir = '../data/raw/'
    csv_paths = [(os.path.abspath(f'{csv_dir}/trial{trial}_fish0.csv'),
                  os.path.abspath(f'{csv_dir}/trial{trial}_fish1.csv')) for trial in trials]
    print(csv_paths)

    dfs_cleaned = {'train': [], 'test': []}
    dfs_kicks = {'train': [], 'test': []}

    for i, (csv_path_0, csv_path_1) in enumerate(csv_paths):
        print(f"Cleaning {csv_path_0} and {csv_path_1}")
        fish_id = tuple(f'f{n}' for n in [i*2, (i*2)+1])
        print(f"fish_id = {fish_id}")

        full_df = load_and_join(csv_path_0, csv_path_1)
        print("Loading and joining done.")

        print(f"Len = {len(full_df)}")
        full_df = clean_dataset(full_df)
        print("Cleaned data.")
        
        # Calculate bounding box for rectangular arena.
        # Important: Do this before interpolating missing frames - doing this afterwards leads to strange bbs!
        bounding_box = (min(full_df['x_f0'].min(), full_df['x_f1'].min()), max(full_df['x_f0'].max(), full_df['x_f1'].max()), 
                        min(full_df['y_f0'].min(), full_df['y_f1'].min()), max(full_df['y_f0'].max(), full_df['y_f1'].max()))
        print(f"Computed bounding box {bounding_box}.")
        
        # Perform train test-split here.
        # TODO: Take first 80% for half of data, final 80% for other half to avoid bias.
        for df, subset_name in zip(cv.train_test_split(full_df, train_size=0.8, test_size=0.2, shuffle=False), ['train', 'test']):
            # Fix index of dataframe (corrupted by splitting.)
            df.index = range(0, len(df))

            print(f"Now considering {subset_name}.")
            df = interpolate_invalid(df)
            df = smooth_dataset(df)
            print("Smoothed velocity and acceleration!")
            df = add_status(df, SWIMMING_THRESHOLD)
            print(f"Stopped frames: {((df['status'] == 'stopping')*1.0).sum()}.")
            print(f"Dropped frames: {((df['dropped'] == True)*1.0).sum()}.")

            print("Found active and passive swimming phases.")
            #print(f"The data-frame has the following columns now:{df.columns}")

            df_kicks = calc_angles_df(df, fish_id, bounding_box)

            print("Calculated angles.")
            #print(f"The kicks_df has the following columns now:{df_kicks.columns}")

            dfs_cleaned[subset_name].append(df.copy())
            print(f"Len cleaned = {len(df)}")
            dfs_kicks[subset_name].append(df_kicks)

    if len(trials) > 1:
        dfs_cleaned = {k: pd.concat(v) for k, v in dfs_cleaned.items()}
        dfs_kicks = {k: pd.concat(v) for k, v in dfs_kicks.items()}

    for subset_name in ['train', 'test']:
        dfs_cleaned[subset_name].to_csv(csv_cleaned.format(subset_name), index=False)
        dfs_kicks[subset_name].to_csv(csv_kicks.format(subset_name), index=False)

if __name__ == '__main__':
    main()
