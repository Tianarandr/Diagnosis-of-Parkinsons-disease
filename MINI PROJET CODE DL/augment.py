import numpy as np
import random
import utils

from dba import calculate_dist_matrix
from dba import dba 
from knn import get_neighbors

# weights calculation method : Average Selected (AS)
def get_weights_average_selected(x_train, dist_pair_mat, distance_algorithm='dtw'):
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
    num_dim = x_train[0].shape[1]
    n = len(x_train)
    max_k = 5 
    max_subk = 2
    k = min(max_k,n-1)
    subk = min(max_subk,k)
    weight_center = 0.5 
    weight_neighbors = 0.3
    weight_remaining = 1.0- weight_center - weight_neighbors
    n_others = n - 1 - subk
    if n_others == 0 : 
        fill_value = 0.0
    else:
        fill_value = weight_remaining/n_others
    idx_center = random.randint(0,n-1)
    init_dba = x_train[idx_center]
    weights = np.full((n,num_dim),fill_value,dtype=np.float64)
    weights[idx_center] = weight_center
    topk_idx = np.array(get_neighbors(x_train,init_dba,k,dist_fun,dist_fun_params,
                         pre_computed_matrix=dist_pair_mat, 
                         index_test_instance= idx_center))
    final_neighbors_idx = np.random.permutation(k)[:subk]
    weights[topk_idx[final_neighbors_idx]] = weight_neighbors / subk
    return weights, init_dba

def augment_train_set(x_train, y_train, classes, N, dba_iters=5, 
                      weights_method_name = 'aa', distance_algorithm='dtw',
                      limit_N = True):
    """
     Cette méthode prend un ensemble de données et l'augmente en utilisant la méthode dans icdm2017.
     :param x_train : le train d'origine
     :param y_train : le jeu d'étiquettes d'origine
     :param N : Le nombre de séries chronologiques synthétiques.
     :param dba_iters : nombre d'itérations dba à faire converger.
     :param weights_method_name : la méthode d'attribution des poids (voir constants.py)
     :param distance_algorithm : Le nom de l'algorithme de distance utilisé (voir constants.py)
    """
    # get the weights function
    weights_fun = utils.constants.WEIGHTS_METHODS[weights_method_name]
    # get the distance function 
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get the distance function params 
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
    # synthetic train set and labels 
    synthetic_x_train = []
    synthetic_y_train = []
    # loop through each class
    print("CLASSSEEEEEEEEEE :", classes)
    for c in classes: 
        # get the MTS for this class 
        c_x_train = x_train[np.where(y_train==c)]

        if len(c_x_train) == 1 :
            # skip if there is only one time series per set
            continue

        if limit_N == True:
            # limit the nb_prototypes
            nb_prototypes_per_class = min(N, len(c_x_train))
        else:
            # number of added prototypes will re-balance classes
            nb_prototypes_per_class = N + (N-len(c_x_train))

        # get the pairwise matrix 
        if weights_method_name == 'aa': 
            # then no need for dist_matrix 
            dist_pair_mat = None 
        else: 
            dist_pair_mat = calculate_dist_matrix(c_x_train,dist_fun,dist_fun_params)
        # loop through the number of synthtectic examples needed
        for n in range(nb_prototypes_per_class): 
            # get the weights and the init for avg method 
            weights, init_avg = weights_fun(c_x_train,dist_pair_mat,
                                            distance_algorithm=distance_algorithm)
            # get the synthetic data 
            synthetic_mts = dba(c_x_train, dba_iters, verbose=False, 
                            distance_algorithm=distance_algorithm,
                            weights=weights,
                            init_avg_method = 'manual',
                            init_avg_series = init_avg)  
            # add the synthetic data to the synthetic train set
            synthetic_x_train.append(synthetic_mts)
            # add the corresponding label 
            synthetic_y_train.append(c)
    # return the synthetic set 
    return np.array(synthetic_x_train), np.array(synthetic_y_train)
            
        
        
    
    

