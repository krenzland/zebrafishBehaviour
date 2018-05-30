import numpy as np
from itertools import product
import dill as pickle
import pandas as pd
import sklearn.preprocessing as pre

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
    dts = np.unique(df['dt'])
    assert(len(df) % len(dts) == 0)
    num_kicks = len(df)//len(dts)
    
    # Each group contains information about a kick and the past data.
    # -> each kick is own group
    groups = np.repeat(np.arange(0, num_kicks), len(dts))
    df['group'] = pd.Series(groups, index=df.index)

    if cutoff_wall_range is not None:
        # Drop all kicks where one timestep is too close to any wall
        wall_dist_col = [f'wall_distance{i}_f{id}' for i, id in product(range(0,4), range(0,2))]
        min_wall_dist = df[wall_dist_col].min(axis=1)

        # Drop kick if fish was influenced by the wall at any point!
        groups_to_keep = np.unique(df.loc[(min_wall_dist > cutoff_wall_range),'group'])
        to_keep = df['group'].isin(groups_to_keep)   
        df = df.loc[to_keep,:]
        df.index = pd.Index(range(0, len(df))) # Fix index

        # Drop unneded columns
    df = df[['dt', 'heading_change', 'x_f0', 'y_f0', 'x_f1', 'y_f1', 'angle_f0', 'angle_f1', 'length', 'group']]
   
    # Rotate coordinate system for each row.
    df = df.apply(transform_coords_row, axis=1)
    
    return df
# Static ------------------------------------------------------------------------------------------------------
def get_bins_static(df, num_bins=7, ignore_outliers=True):
    # Build symmetric receptive field.
    if ignore_outliers:
        lo_x, hi_x = np.percentile(a=df_train['x_f1'], q=[25,75])
        lo_y, hi_y = np.percentile(a=df_train['x_f1'], q=[25,75])
        iqr_x = hi_x - lo_x
        iqr_y = hi_y - lo_y

        # Bin range without outliers (using 1.5 IQR)
        lower_limit = min(lo_x - 1.5*iqr_x, lo_y - 1.5*iqr_y)
        upper_limit = max(hi_x + 1.5*iqr_x, hi_y + 1.5 * iqr_y)

        rf_size = max(-lower_limit, upper_limit)        
    else:
        rf_size = max(df['x_f1'].abs().max(), df['y_f1'].abs().max())
    print(rf_size)
    
    # RF has size rf_size x rf_size, each direction divided by num_bins
    # Should be of form 2n-1 (symmetric in pos/neg. direction, centered at (0,0))
    #assert(((num_bins-1) % 2) == 0)
    
    b = np.linspace(rf_size, 0, num=num_bins//2, endpoint=False)
    #b = np.array([rf_size/(i+1) for i in range(num_bins//2)])
    bins = np.hstack((-b,[0], b[::-1]))
    
    return bins

def digitize_df_static(df, bins, closed_interval=True):
    # TODO: Open intervals on both sides!
    bin_x = np.digitize(df['x_f1'], bins=bins) - 1
    bin_y = np.digitize(df['y_f1'], bins=bins) - 1
    
    print('Before clipping', bin_x.min(), bin_y.min(), bin_x.max(), bin_y.max())
    # Clip to range
    bin_x = bin_x.clip(0, num_bins-1)
    bin_y = bin_y.clip(0, num_bins-1)
    print('After clipping', bin_x.min(), bin_y.min(), bin_x.max(), bin_y.max())

    print(bin_x.min(), bin_x.max(), bin_y.max())
    
    bin = (bin_x*num_bins) + bin_y
    if closed_interval:
        # Fish outside range have bin -1
        bin[(df['x_f1'] <= bins.min()) | (df['y_f1'] <= bins.min()) |
           (df['x_f1'] >= bins.max()) | (df['y_f1'] >= bins.max())] = -1
    
    return bin


# Adaptive ----------------------------------------------------------------------------------------------------
def bin_one_dim(array, num_bins):
# Inspired by np.array_split
# https://github.com/numpy/numpy/blob/v1.14.0/numpy/lib/shape_base.py#L431-L483
# Divide into equal bins, if not possible first few bins have size + 1
    
    a = np.sort(array)
    assert(np.signbit(a[0]) == np.signbit(a[-1]))
    num_per_bin, extras = divmod(len(a), num_bins)
    section_sizes = [0] + extras * [num_per_bin+1] + (num_bins-extras) * [num_per_bin]
    div_ind = np.array(section_sizes).cumsum()
    div_ind[-1] -= 1
    div_ind, len(a)
    bin_edges = a[div_ind]
    if (np.signbit(a[0])):
        bin_edges[-1] = 0.0
    else:
        bin_edges[0] = 0.0
    return bin_edges

# Try without symmetry
def bin_axis(array, num_bins):
    assert(num_bins % 2 == 0)
    #num_bins += 1
    dist_pos = array[array >= 0]
    dist_neg = array[array < 0]
    edges_pos = bin_one_dim(dist_pos, num_bins//2)
    edges_neg = bin_one_dim(dist_neg, num_bins//2)#[::-1]
    edges = np.hstack((edges_neg, edges_pos[1:]))
    
    # Fix empty bins on either side (make them bit larger than data range)
    eps = 10e-6
    edges[0] -= eps
    edges[-1] += eps
    return edges

def bin_df(df, num_bins=6):
    edges_x = bin_axis(df['x_f1'].values, num_bins)
    bins_x = np.digitize(df['x_f1'].values, edges_x, right=False) - 1
    #bins_y = np.zeros_like(bins_x) - 1 # invalid bins for now!

    edges_y = np.zeros((num_bins, num_bins+1))

    for i in range(num_bins):
        is_in_bin = bins_x == i
        cur_y = df.loc[is_in_bin,'y_f1'].values
        cur_edges_y = bin_axis(cur_y, num_bins)
        edges_y[i,:] = cur_edges_y

    return edges_x, edges_y

def get_bins_df(df, edges_x, edges_y, num_bins):
    bins_x = np.digitize(df['x_f1'].values, edges_x, right=False) - 1
    bins_x = bins_x.clip(0, len(edges_x) - 2) # todo
    bins_y = np.zeros_like(bins_x) - 1 # invalid bins for now!

    for i in range(num_bins):
        is_in_bin = bins_x == i
        #cur_y = df.loc[is_in_bin,'y_f1'].values
        #cur_edges_y = bin_axis(cur_y, num_bins)
        #edges_y[i,:] = cur_edges_y
        cur_bins_y = np.digitize(df.loc[is_in_bin, 'y_f1'].values,
                                 edges_y[i,:],
                                 right=False) - 1
        cur_bins_y = cur_bins_y.clip(0, len(edges_x) - 2) # todo
        bins_y[is_in_bin] = cur_bins_y

    bins = (bins_x*num_bins) + bins_y
    df.loc[:,'bin'] = pd.Series(bins, index=df.index)    
    
    return df

def get_Xy(df, num_bins, means, stds):
    # Now we need to one-hot enccode our data.
    # We need one column per variable and bin

    # Indicator for position
    # We need this because some bins could be unoccupied in training but occupied in testing!
    one_hot_enc = pre.OneHotEncoder()
    one_hot_enc.fit(np.array(list(range(0,num_bins**2))).reshape(-1,1))
    
    position_one_hot = one_hot_enc.transform(df.loc[0:, 'bin'].values.reshape(-1,1)).toarray()
    
    # Standarize x and y trajectories    
    direction_f0_x = df['trajectory_f1_x'].values.reshape(-1, 1)
    direction_f0_x = (direction_f0_x - means[0]) / stds[0]
    direction_f0_x = position_one_hot * direction_f0_x
    
    direction_f0_y = df['trajectory_f1_y'].values.reshape(-1, 1)
    direction_f0_y = (direction_f0_y - means[1]) / stds[1]
    direction_f0_y = position_one_hot * direction_f0_y

    dts =  df['dt'].values.reshape(-1, 1)

    X = np.concatenate((dts, position_one_hot, direction_f0_x, direction_f0_y), axis=1)
    #X = position_one_hot.values
    y = np.vstack((df['trajectory_f0_x'].values.T, df['trajectory_f0_y'].values.T)).T
    return X, y

def save_csv(X, y, file_name=None):
    processed_df = Xy_to_df(X,y)
    processed_df.to_csv(file_name, index=None)


def Xy_to_df(X,y):
    num_dts = X.shape[0]//y.shape[0]
    y = np.repeat(y, num_dts, axis=0)
    processed_df = pd.DataFrame(np.vstack((X.T, y.T)).T)
    cols = ['dt'] + [f'feature_{i}' for i in range(0, X.shape[1] - 1)] + ['y_0', 'y_1']
    processed_df.columns = cols    
    return processed_df


def main():
    df_kicks_tr = pd.read_csv('../data/processed/kicks_guy_train.csv')
    df_kicks_te = pd.read_csv('../data/processed/kicks_guy_test.csv')
    print("Loaded data.")

    # Determine cutoff range for wall forces.
    with open('../models/calovi_wall.model', mode='rb') as f:
        wall_model = pickle.load(f)

    cutoff_wall_range = wall_neglible_at(wall_model)
    print(f"Using cutoff wall range of {cutoff_wall_range}.")

    # Compute relative coords/angles
    df_train = transform_coords_df(df_kicks_tr.copy(), cutoff_wall_range=cutoff_wall_range)
    df_test = transform_coords_df(df_kicks_te.copy(), cutoff_wall_range=cutoff_wall_range) 
    print("Computed relative coordinates.")

    num_bins = 8
    edges_x, edges_y = bin_df(df_train, num_bins=num_bins)

    edges = {'num_bins': num_bins,
            'edges_x': edges_x,
            'edges_y': edges_y}
    with open('../models/adaptive_bins.model', 'wb') as f:
        pickle.dump(edges, f)

    print("Saved bin edges.")

    df_train = get_bins_df(df_train, **edges)
    df_test = get_bins_df(df_test, **edges) 
    print("Computed spatial discretization")
    
    means = df_train['trajectory_f1_x'].mean(), df_train['trajectory_f1_y'].mean()
    stds = df_train['trajectory_f1_x'].std(), df_train['trajectory_f1_y'].std()
    print(means, stds)

    print(df_train.columns)
    X_np, y_np = get_Xy(df_train, num_bins=num_bins, means=means, stds=stds)
    X_np_test, y_np_test = get_Xy(df_test, num_bins=num_bins, means=means, stds=stds)

    y_np =  y_np[df_train['dt'] == 0] # heading change at kick
    y_np_test =  y_np_test[df_test['dt'] == 0] # heading change at kick

    print(X_np.shape, X_np_test.shape)

    save_csv(X_np, y_np, '../data/processed/rf_train.csv')
    save_csv(X_np_test, y_np_test, '../data/processed/rf_test.csv')
    print("Saved csv.")


if __name__ == '__main__':
    main()
