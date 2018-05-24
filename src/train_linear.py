import numpy as np
import dill as pickle
from sklearn.linear_model import RidgeCV
from train_mdn import get_data

def convert_data_concatenate(X):
   num_dts = X.shape[0]
   num_features = X.shape[2] 
   return X.transpose(1,0,2).reshape(-1, num_dts * num_features)

def convert_data_no_memory(X):
    return X[-1,:,:]

def fit_linear(X, y, lambda_range):
    model = RidgeCV(alphas=lambda_range)
    model.fit(X, y)
    return model

def main():
    print("Loading data.")
    X_train, y_train, X_test, y_test = [d.numpy() for d in get_data('../data/processed')]

    X_train_conc = convert_data_concatenate(X_train)
    X_test_conc = convert_data_concatenate(X_test)

    X_train_no_memory = convert_data_no_memory(X_train)
    X_test_no_memory = convert_data_no_memory(X_test)

    lambda_range =np.logspace(-3, 4, num=128) 
    
    print("Loaded data.")
    print("Fitting concatenate.")
    linear_concatenate = fit_linear(X_train_conc, y_train, lambda_range)
    print("Fitted concatenate.")

    print("Fitting no memory.")
    linear_no_memory = fit_linear(X_train_no_memory, y_train, lambda_range)
    print("Fitted no memory.")

    print("Saving models.")
    with open('../models/linear_concatenate.model', 'wb') as file:
        pickle.dump(linear_concatenate, file)
    
    with open('../models/linear_no_memory.model', 'wb') as file:
        pickle.dump(linear_no_memory, file)
    print("Saved models.")

if __name__ == '__main__':
    main()
