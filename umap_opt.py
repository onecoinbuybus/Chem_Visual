import numpy as np
from scipy.spatial import distance
import optuna
import umap


def k3nerror(X1, X2, k):
    X1 = np.array(X1)
    X2 = np.array(X2)

    X1_dist = distance.cdist(X1, X1)
    X1_sorted_indices = np.argsort(X1_dist, axis=1)
    X2_dist = distance.cdist(X2, X2)

    for i in range(X2.shape[0]):
        _replace_zero_with_the_smallest_positive_values(X2_dist[i, :])

    I = np.eye(len(X1_dist), dtype=bool)
    neighbor_dist_in_X1 = np.sort(X2_dist[:, X1_sorted_indices[:, 1:k+1]][I])
    neighbor_dist_in_X2 = np.sort(X2_dist)[:, 1:k+1]

    sum_k3nerror = (
            (neighbor_dist_in_X1 - neighbor_dist_in_X2) / neighbor_dist_in_X2
           ).sum()
    return sum_k3nerror / X1.shape[0] / k

def _replace_zero_with_the_smallest_positive_values(arr):
    arr[arr==0] = np.min(arr[arr!=0])

def k3nerror_output(min_dist, fingerprint):
    k_in_k3nerror=10
    Z_in_min_dist_optimization = umap.UMAP(n_neighbors=10, 
                                             min_dist=min_dist, 
                                             n_components=2, 
                                             metric='euclidean',
                                             random_state=0).fit_transform(fingerprint)
    
    scaled_Z_in_min_dist_optimization = (Z_in_min_dist_optimization - Z_in_min_dist_optimization.mean(
        axis=0)) / Z_in_min_dist_optimization.std(axis=0, ddof=1)
    
    k3nerror_output=(k3nerror(fingerprint, scaled_Z_in_min_dist_optimization, k_in_k3nerror) + k3nerror(
            scaled_Z_in_min_dist_optimization, fingerprint, k_in_k3nerror))
    
    return k3nerror_output

def objective(trial, fingerprint, params_dict):
    min_dist = trial.suggest_uniform('min_dist', params_dict['min_dist'][0], params_dict['min_dist'][1])
    n_neighbors = trial.suggest_int('n_neighbors', params_dict['n_neighbors'][0], params_dict['n_neighbors'][1])

    Z = umap.UMAP(n_neighbors=n_neighbors, 
                 min_dist=min_dist, 
                 n_components=2, 
                 metric='euclidean',
                 random_state=0).fit_transform(fingerprint)
    
    scaled_Z = (Z - Z.mean(axis=0)) / Z.std(axis=0, ddof=1)
    k3nerror_value = -(k3nerror(fingerprint, scaled_Z, n_neighbors) + k3nerror(scaled_Z, fingerprint, n_neighbors))
    
    return k3nerror_value

def optimize_hyperparams(params_dict, fingerprint, n_trials):
    study = optuna.create_study()
    func = lambda trial: objective(trial, fingerprint, params_dict)
    study.optimize(func, n_trials=n_trials)

    best_params = study.best_params
    return best_params