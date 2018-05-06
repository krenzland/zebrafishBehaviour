import numpy as np
from itertools import product
import pandas as pd

from util import add_angles, angle_between, angled_vector, sub_angles, get_rotation_matrix
from calovi import WallModel

from scipy import optimize as opt

def wall_neglible_at(wall_model, threshold=0.1):
    """Finds minimal wall distance s.t. wall force is smaller than threshold."""
    def f(x):
        return wall_model.wall_force(x) - threshold
    return opt.brentq(f, a=0, b=30)


def transform_coords_df(df, cutoff_wall_range):
    """Rotates coordinate system s.t. fish zero is at (0,0) and looking to the right.
       Drops kicks close to the wall."""
    def transform_coords_row(row):
        rotation_angle = row['angle_f0']
        rotation_matrix = get_rotation_matrix(-rotation_angle)

        pos_f0 = np.array([row['x_f0'], row['y_f0']])
        pos_f1 = np.array([row['x_f1'], row['y_f1']])

        # Rotate both fish s.t. fish 0 has angle 0
        pos_f0 = (rotation_matrix @ pos_f0.reshape(2,1))
        pos_f1 = (rotation_matrix @ pos_f1.reshape(2,1))

        # Put fish zero at (0,0) in coord system
        pos_f1 = (pos_f1 - pos_f0).reshape(2)
        pos_f0 = np.array([0,0])

        angle_f1 = sub_angles(row['angle_f1'], rotation_angle)

        trajectory_f0 = angled_vector(row['heading_change']) * row['length']
        trajectory_f1 = angled_vector(angle_f1)

        row = pd.Series(data=[row['dt'], trajectory_f0[0], trajectory_f0[1], pos_f1[0], pos_f1[1], 
                            trajectory_f1[0], trajectory_f1[1]],
                        index=['dt', 'trajectory_f0_x', 'trajectory_f0_y', 'x_f1', 'y_f1',
                            'trajectory_f1_x', 'trajectory_f1_y'])
        return row

    # Group columns by kick (dt=0,1,..)
    #max_dt = df['dt'].max() + 1
    
    dts = np.unique(df['dt'])
    assert(len(df) % len(dts) == 0)
    num_kicks = len(df)//len(dts)
    
    # Each group contains information about a kick and the past data.
    # -> each kick is own group
    groups = np.repeat(np.arange(0, num_kicks), len(dts))
    df['group'] = pd.Series(groups, index=df.index)

    # Drop all kicks where one timestep is too close to any wall
    wall_dist_col = [f'wall_distance{i}_f{id}' for i, id in product(range(0,4), range(0,2))]
    min_wall_dist = df[wall_dist_col].min(axis=1)

    # Drop kick if fish was influenced by the wall at any point!
    # TODO: Maybe only drop if kick happened next to the wall?
    groups_to_keep = np.unique(df.loc[(min_wall_dist > cutoff_wall_range),'group'])
    to_keep = df['group'].isin(groups_to_keep)   
    df = df.loc[to_keep,:]
    df.index = pd.Index(range(0, len(df))) # Fix index

    # Drop unneded columns
    df = df[['dt', 'heading_change', 'x_f0', 'y_f0', 'x_f1', 'y_f1', 'angle_f0', 'angle_f1', 'length', 'group']]
   
    # Rotate coordinate system for each row.
    df = df.apply(transform_coords_row, axis=1)
    
    return df


