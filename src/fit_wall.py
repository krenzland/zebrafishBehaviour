#!/usr/bin/env python3

import numpy as np
import pandas as pd
import dill as pickle
from scipy.optimize import curve_fit
from scipy.stats import spearmanr

from calovi import WallModel
from util import add_angles

def get_data_wall(data):  
    heading_change = data['heading_change'].values
    wall_distance = np.vstack([data[f'wall_distance{i}_f0'].values for i in range(4)])
    wall_angle = np.vstack([data[f'wall_angle{i}_f0'].values for i in range(4)])
    xdata_wall = np.vstack((wall_distance, wall_angle))
    ydata = heading_change

    return xdata_wall, ydata

def calc_wall_error(model, xdata, ydata):
    return np.linalg.norm(model(xdata) - ydata)/(len(ydata))

def compute_r2(x, y, model=None, yhat=None):
    if yhat is None:
        yhat = model(x)
    residuals = y - yhat
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    return 1 - (ss_res/ss_tot)

def evaluate_model(data, wall_model,  x_opt_wall):
    xdata_wall, ydata = get_data_wall(data)
    fitted_model = lambda xdata: wall_model.evaluate_raw(xdata, *x_opt_wall)
    random_model = lambda xdata: wall_model.evaluate_raw(xdata, *x0)
    mean_model = lambda xdata: np.ones_like(xdata) * np.mean(ydata)

    print(f"Curve fit reached error of {calc_wall_error(fitted_model, xdata_wall, ydata)} with params={str(x_opt_wall)}")

    # Calculate error for mean heading change as model
    mean_heading_error = calc_wall_error(mean_model, xdata_wall, ydata)
    print(f"Predicting the average heading {np.mean(ydata)} for each heading leads to an error of {mean_heading_error}")
    mean_angle = np.arctan2(np.sin(ydata).sum(),np.cos(ydata).sum())
    print(mean_angle)
    
    print(f"Model has r^2 of {compute_r2(xdata_wall, ydata, fitted_model)}")
    yHat = fitted_model(xdata_wall)
    print(f"{spearmanr(ydata, -fitted_model(xdata_wall))}")
    
    print(f'Cos-Error-Model: {(1 - np.cos(ydata - yHat)).sum()}')
    print(f'Cos-Error-Mean: {(1 - np.cos(ydata - mean_angle)).sum()})')
    
    print(f'Arccos-Error-Model: {np.arccos(np.cos(ydata - yHat)).mean()}')
    print(f'Arccos-Error-Mean: {np.arccos(np.cos(ydata - mean_angle)).mean()})')  

def fit_wall(data):
    # Setup data.
    xdata_wall, ydata = get_data_wall(data)

    # Initialise model with standard parameters.
    angular_model = 'sin-cos'
    wall_model = WallModel(angular_model)
    angular = WallModel.angular_map
    init_x = WallModel.params_map

    # Setup helper methods and bounds for optimizer.
    x0 = init_x[angular_model]

    # Fit the model
    res = curve_fit(wall_model.evaluate_raw, xdata=xdata_wall, ydata=ydata, p0=x0)
    x_opt_wall = res[0]
    
    wall_model.set_params(x_opt_wall)
    return wall_model, x0, x_opt_wall
                                                 

def main():
    train = pd.read_csv("../data/processed/kicks_guy_train.csv")
    test = pd.read_csv("../data/processed/kicks_guy_test.csv")
    print("Loaded data.")

    train = train.loc[ train['dt'] == 0]
    test = test.loc[ test['dt'] == 0]

    
    wall_model, x0, x_opt_wall = fit_wall(train)
    print("Fitted model.")

    with open('../models/calovi_wall.model', 'wb') as file:
        pickle.dump(wall_model, file)
    print("Saved model")

    print("Evaluating training")
    evaluate_model(train, wall_model, x_opt_wall)

    print('\n', '-'*40)
    print("Evaluating testing")
    evaluate_model(test, wall_model, x_opt_wall)

if __name__ == '__main__':
    main()
