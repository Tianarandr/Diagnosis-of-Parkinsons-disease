import numpy as np 
import utils 

def calculate_dist_matrix(tseries, dist_fun, dist_fun_params):
    N = len(tseries)
    pairwise_dist_matrix = np.zeros((N,N), dtype = np.float64)
    # pre-compute the pairwise distance
    for i in range(N-1):
        x = tseries[i]
        for j in range(i+1,N):
            y = tseries[j] 
            dist = dist_fun(x,y,**dist_fun_params)[0] 
            # because dtw returns the sqrt
            dist = dist*dist 
            pairwise_dist_matrix[i,j] = dist 
            # dtw is symmetric 
            pairwise_dist_matrix[j,i] = dist 
        pairwise_dist_matrix[i,i] = 0 
    return pairwise_dist_matrix

def medoid(tseries, dist_fun, dist_fun_params):
    """
     Calcule le médioïde de la liste donnée de MTS
     :param tseries : La liste des séries temporelles
    """
    N = len(tseries)
    if N == 1 : 
        return 0,tseries[0]
    pairwise_dist_matrix = calculate_dist_matrix(tseries, dist_fun, 
                                                 dist_fun_params)
        
    sum_dist = np.sum(pairwise_dist_matrix, axis = 0)
    min_idx = np.argmin(sum_dist)
    med = tseries[min_idx]
    return min_idx, med

def _dba_iteration(tseries, avg, dist_fun, dist_fun_params,weights):
    """
     Effectuez une itération dba pondérée et renvoyez la nouvelle moyenne
     """
    # the number of time series in the set
    n = len(tseries)
    # length of the time series 
    ntime = avg.shape[0]
    # number of dimensions (useful for MTS)
    num_dim = avg.shape[1]
    # array containing the new weighted average sequence 
    new_avg = np.zeros((ntime,num_dim),dtype=np.float64) 
    # array of sum of weights 
    sum_weights = np.zeros((ntime,num_dim),dtype=np.float64)
    # loop the time series 
    for s in range(n): 
        series = tseries[s]
        dtw_dist, dtw = dist_fun(avg, series, **dist_fun_params)
        
        i = ntime 
        j = series.shape[0]
        while i >= 1 and j >= 1:
            new_avg[i-1] += series[j-1]*weights[s]
            sum_weights[i-1] += weights[s]
            
            a = dtw[i - 1, j - 1]
            b = dtw[i, j - 1]
            c = dtw[i - 1, j]
            if a < b:
                if a < c:
                    # a is the minimum
                    i -= 1
                    j -= 1
                else:
                    # c is the minimum
                    i -=1 
            else:
                if b < c:
                    # b is the minimum
                    j -= 1
                else:
                    # c is the minimum
                    i -= 1
    # update the new weighted avgerage 
    new_avg = new_avg/sum_weights
    
    return new_avg
        
def dba(tseries, max_iter =10, verbose=False, init_avg_method = 'medoid', 
        init_avg_series = None, distance_algorithm = 'dtw', weights=None): 
    """
    Calcule la moyenne du barycentre (DBA) de déformation temporelle dynamique (DTW) d'un
    groupe de séries temporelles multivariées (MTS).
    :param tseries : une liste contenant les séries à moyenner, où chaque
        MTS a une forme (l,m) où l est la longueur de la série chronologique et
        m est le nombre de dimensions du MTS - dans le cas de univarié
        la série temporelle m doit être égale à un
    :param max_iter : Le nombre maximum d'itérations pour l'algorithme DBA.
    :param verbose : si vrai, fournit une sortie utile.
    :param init_avg_method : Soit :
        'aléatoire' la moyenne sera initialisée par une série temporelle aléatoire,
        'medoid'(default) la moyenne sera initialisée par la medoid de tseries,
        'manual' la valeur dans init_avg_series sera utilisée pour initialiser la moyenne
    :param init_avg_series : cela sera considéré comme une initialisation moyenne si
        init_avg_method est défini sur 'manuel'
    :param distance_algorithm : détermine la distance à utiliser lors de l'alignement
        la série chronologique
    :param weights : un tableau contenant les poids pour calculer un dba pondéré
        (NB : pour MTS, chaque dimension doit avoir son propre ensemble de pondérations)
        la forme attendue est (n,m) où n est le nombre de séries chronologiques dans tseries
        et m est le nombre de dimensions
    """
    # get the distance function 
    dist_fun = utils.constants.DISTANCE_ALGORITHMS[distance_algorithm]
    # get the distance function params 
    dist_fun_params = utils.constants.DISTANCE_ALGORITHMS_PARAMS[distance_algorithm]
    # check if given dataset is empty 
    if len(tseries)==0: 
        # then return a random time series because the average cannot be computed 
        start_idx = np.random.randint(0,len(tseries))
        return np.copy(tseries[start_idx])
    
    # init DBA
    if init_avg_method == 'medoid':
        avg = np.copy(medoid(tseries,dist_fun, dist_fun_params)[1])
    elif init_avg_method == 'random': 
        start_idx = np.random.randint(0,len(tseries))
        avg = np.copy(tseries[start_idx])
    else: # init with the given init_avg_series
        avg = np.copy(init_avg_series)
        
    if len(tseries) == 1:
        return avg
    if verbose == True: 
        print('Doing iteration')
        
    # main DBA loop 
    for i in range(max_iter):
        if verbose == True:
            print(' ',i,'...')
        if weights is None:
            # when giving all time series a weight equal to one we have the 
            # non - weighted version of DBA 
            weights = np.ones((len(tseries),tseries[0].shape[1]), dtype=np.float64)
        # dba iteration 
        avg = _dba_iteration(tseries,avg,dist_fun, dist_fun_params,weights)
    
    return avg 
    