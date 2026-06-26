## LIBRARIES -------------------------------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import fathon
from fathon import fathonUtils as fu
import time
import scipy

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import root_mean_squared_error as rmse, mean_squared_error as mse, mean_absolute_percentage_error as mape, root_mean_squared_log_error as rmsle

from teaspoon.parameter_selection.FNN_n import FNN_n
from neurokit2.complexity.information_fisher import fisher_information
from nolds import hurst_rs as nolds__hurst
from antropy import svd_entropy as ant__svd
from fathon import DFA


import warnings

## Classes ---------------------------------------------------------------

class map_1d:
    def __init__(self, name, param_dict, required_params):
        for required_key in required_params:
            if required_key not in param_dict:
                raise Exception("Missing parameter: {}".format(required_key))
        for provided_param in param_dict:
            if provided_param not in required_params:
                warnings.warn("Warning: Extra parameter provided: {}".format(provided_param))

        self.name = str(name)
        self.params = {key: param_dict[key] for key in required_params}

    def __str__(self):
        return self.name
    
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.params == other.params:
                return True
        return False
    
    def info(self):
        print(self.name)
        for param_key in self.params:
            print(param_key, "=", self.params[param_key])

class log_map(map_1d):
    def __init__(self, param_dict):
        map_1d.__init__(self, "log_map", param_dict, required_params=["a"])
        self.a = self.params["a"]

    def step(self, x):
        return self.a*x*(1-x)
    
    def deriv(self, x):
        return self.a*(1-(2*x))
    
class sin_map(map_1d):
    def __init__(self, param_dict):
        map_1d.__init__(self, "sin_map", param_dict, required_params=["a"])
        self.a = self.params["a"]

    def step(self, x):
        return self.a*np.sin(np.pi*x, dtype=np.float64)
    
    def deriv(self, x):
        return np.pi*self.a*np.cos(np.pi*x, dtype=np.float64)
    
class tent_map(map_1d):
    def __init__(self, param_dict):
        map_1d.__init__(self, "tent_map", param_dict, required_params=["a", "b"])
        self.a = self.params["a"]
        self.b = self.params["b"]

    def step(self, x):
        return self.a*np.minimum((self.b*x) % 1, ((1-self.b*x)) % 1, dtype=np.float64)
    
    def deriv(self, x):
        return np.where((self.b*x) % 1 < (1-self.b*x) % 1, self.a*self.b, -self.a*self.b)
    
class chop_map(map_1d):
    def __init__(self, param_dict):
        map_1d.__init__(self, "chop_map", param_dict, required_params=["a"])
        self.a = self.params["a"]

    def step(self, x):
        return (self.a*x) % 1
    
    def deriv(self, x):
        return self.a * np.ones(x.shape, dtype=np.float64)

class sluze_map(map_1d):
    def __init__(self, param_dict):
        map_1d.__init__(self, "sluze_map", param_dict, required_params=["m", "p"])
        self.m = self.params["m"]
        self.p = self.params["p"]

    def step(self, x):
        return ((self.m+self.p)**(self.m+self.p))/((self.m**self.m)*(self.p**self.p))*(x**self.p)*((1-x)**self.m)
    
    def deriv(self, x):
        return (((self.p+self.m)**(self.p+self.m))*((1-x)**self.m)*(x**(self.p-1))*(x*(self.p + self.m) - self.p))/((self.m**self.m)*(self.p**self.p)*(x-1))

class sincircle_map(map_1d):
    def __init__(self, param_dict):
        map_1d.__init__(self, "sin_circle_map", param_dict, required_params=["omega", "k"])
        self.k = self.params["k"]
        self.omega = self.params["omega"]

    def step(self, x):
        return (x + self.omega - (self.k/(2*np.pi))*np.sin(2*np.pi* x)) % 1
    
    def deriv(self, x):
        return 1 - self.k*np.cos(2*np.pi*x)

class CML:
    def __init__(self, matrix, map_list):
        assert len(matrix.shape) == 2
        assert matrix.shape[0] == matrix.shape[1]
        assert matrix.shape[0] == len(map_list) 

        rowsums = np.sum(matrix, axis=1)
        assert np.all(np.isclose(rowsums, 1, atol = (10**(-12))))

        self.dim = len(map_list)
        self.matrix = matrix
        self.maps = map_list

    def __eq__(self, other):
        if isinstance(other, CML):
            same_mat = np.array_equal(self.matrix, other.matrix)
            same_maps = self.maps == other.maps
            return same_mat and same_maps
        return False

    def map_info(self):
        for idx, mp in enumerate(self.maps):
            print("map {}:".format(idx+1), mp.name, mp.params)

    def step(self, x):
        fx = np.empty(x.shape, dtype=np.float64)
        for idx, mp in enumerate(self.maps):
            fx[idx] = mp.step(x[idx])
        return np.matmul(self.matrix, fx, dtype=np.float64)
    
    def jacobian(self, x):
        fprime = np.empty(self.dim, dtype=np.float64)
        jacobs = np.empty((self.dim, self.dim), dtype=np.float64)
        for idx, mp in enumerate(self.maps):
            fprime[idx] = mp.deriv(x[idx])[0]

        for i in range(self.dim):
            jacobs[i] = self.matrix[i] * fprime
            # for j in range(self.dim):
            #     jacobs[i][j] = self.matrix[i][j]*fprime[j][0]
        return jacobs

## CML Functions ---------------------------------------------------------

def is_base2(x): # user for hypercube matrix
    root = np.log2(x)
    return x == 2**np.floor(root)

def row_norm(matrix): #used for random matrix
    rowsums = np.sum(matrix, axis=1)
    if np.all(rowsums == 1):
        return matrix
    
    else:
        matrix = np.round(matrix, 5)
        for i in range(matrix.shape[0]):
            matrix[i] = matrix[i]/rowsums[i]
            
        rowsums = np.sum(matrix, axis=1)
        if np.all(np.isclose(rowsums, 1)):
            return matrix
        else:
            warnings.warn("Rowsums not unity: {}".format(rowsums))
            return matrix
        
def full_connected_mat(dim, epsilon, if_print=False, **kwargs):

    if not (epsilon <= 1 and epsilon >= 0):
        raise Exception("Invalid epsilon for non-random matrix")
    
    if dim == 1:
        warnings.warn("dim = 1, returning [[1]]")
        return np.ones((1,1))
    
    diag = 1-epsilon
    offdiag = epsilon/(dim-1)
    A = diag*np.eye(dim, dtype=np.float64) + offdiag*np.ones((dim,dim), dtype=np.float64) - offdiag*np.eye(dim, dtype=np.float64)
    A = np.reshape(A, (dim, dim))
    if if_print:
        print(A)
    
    assert np.all(np.logical_and(A >= 0, A <= 1))
    assert np.all(np.isclose(np.sum(A, axis=1), 1))
    return A
    
def nearest_neigh_mat(dim, epsilon, if_print=False, **kwargs):

    if not (epsilon <= 1 and epsilon >= 0):
        raise Exception("Invalid epsilon for non-random matrix")
    
    if dim == 1:
        warnings.warn("dim = 1, returning [[1]]")
        return np.ones((1,1))
    A = np.zeros((dim,dim), dtype=np.float64)
    for i in range(dim):
        A[i][i] = 1-epsilon
        A[i][i-1] += epsilon/2 
        A[i][(i+1)%dim] += epsilon/2
        ##by adding epsilon/2 each time, we ensure the matrix is still normalised for the 2D case
    A = np.reshape(A, (dim, dim))
    
    if if_print:
        print(A)
        
    assert np.all(np.logical_and(A >= 0, A <= 1))
    assert np.all(np.isclose(np.sum(A, axis=1), 1))
    return A

def fc_decay_mat(dim, epsilon, gamma=1, if_print=False, **kwargs): ## https://doi.org/10.1007/s11071-016-3135-0

    if not (epsilon <= 1 and epsilon >= 0):
        raise Exception("Invalid epsilon for non-random matrix")
    
    if dim == 1:
        warnings.warn("dim = 1, returning [[1]]")
        return np.ones((1,1))
    
    if dim % 2 == 1:
        is_odd = True
    else:
        is_odd = False
    
    A = np.zeros((dim,dim), dtype=np.float64)

    if is_odd:
        k = 0
        for s in range(1, int(((dim-1)/2)+1)):
            k += 2*np.exp(-s*gamma)
    else:
        k = np.exp(-gamma*dim/2)
        for s in range(1, int(((dim-2)/2)+1)):
            k += 2*np.exp(-s*gamma)
        
    for i in range(dim):
        for j in range(dim):
            dist = min(abs(i-j), abs(dim-i+j), abs(dim-j+i))
            if dist == 0:
                A[i,j] = 1 - epsilon
            else:
                A[i,j] = (epsilon/k)*np.exp(-dist*gamma)

    if if_print:
        print(A)

    assert np.all(np.logical_and(A >= 0, A <= 1))
    assert np.all(np.isclose(np.sum(A, axis=1), 1))
    return A

def hypercube_mat(dim, epsilon, if_print=False, **kwargs):

    if not (epsilon <= 1 and epsilon >= 0):
        raise Exception("Invalid epsilon for non-random matrix")
    
    if dim == 1:
        warnings.warn("dim = 1, returning [[1]]")
        return np.ones((1,1))
        
    root = np.log2(dim)
    
    if not is_base2(dim):
        raise Exception("dim is not 2**N \n dim = {}".format(dim))

    root = int(root)

    adj_1 = np.array([[0,1],[1,0]])
    adj = adj_1
    for i in range(2, root+1):
        adj = (np.kron(np.identity(2, dtype=np.float64), adj)) + (np.kron(adj_1, np.identity(2**(i-1), dtype=np.float64)))
    
    A = (epsilon/root)*adj + (1-epsilon)*np.identity(2**(root), dtype=np.float64)
    
    A = np.reshape(A, (dim, dim))
    
    if if_print:
        print(A)

    assert np.all(np.logical_and(A >= 0, A <= 1))
    assert np.all(np.isclose(np.sum(A, axis=1), 1))
    return A
    
def random_mat(dim, seed=1, if_print=False, **kwargs):
    if dim == 1:
        warnings.warn("dim = 1, returning [[1]]")
        return np.ones((1,1))
    np.random.seed(seed)
    A = np.round(np.random.rand(dim, dim), 5)
    A = row_norm(A)
    assert np.all(np.logical_and(A >= 0, A <= 1))
    assert np.all(np.isclose(np.sum(A, axis=1), 1))
    if if_print:
        print(A)
    return A

def one_way_matrix(dim, epsilon, reverse=False, if_print=False, **kwargs):
    
    if not (epsilon <= 1 and epsilon >= 0):
        raise Exception("Invalid epsilon for non-random matrix")
    
    if dim == 1:
        warnings.warn("dim = 1, returning [[1]]")
        return np.ones((1,1))

    A = np.zeros((dim,dim), dtype=np.float64)
    
    for i in range(dim):
        A[i][i] = 1-epsilon
        if reverse:
            A[i][i-1] += epsilon
        else:
            A[i][(i+1)%dim] += epsilon
    A = np.reshape(A, (dim, dim))
    
    if if_print:
        print(A)
        
    assert np.all(np.logical_and(A >= 0, A <= 1))
    assert np.all(np.isclose(np.sum(A, axis=1), 1))
    return A

def gen_matrix(typ, d, if_print=False, **kwargs):
    t = typ.upper().strip()

    if t not in ['RAND', 'NN', 'FC', 'FCD', 'OW', 'HC']:
        raise Exception("Invalid matrix type: typ={}".format(typ))
    
    if d < 1 or int(d) != d:
        raise Exception("Invalid dimension: d={}".format(d))

    if d == 1:
        m = np.ones((1,1))
        if if_print:
            print(m)
        return m

    if t == "FC":
        m = full_connected_mat(dim=d, **kwargs, if_print=if_print)
    elif t == "NN":
        m = nearest_neigh_mat(dim=d, **kwargs, if_print=if_print)
    elif t == "FCD":
        m = fc_decay_mat(dim=d, **kwargs, if_print=if_print)
    elif t == "HC":
        if is_base2(d):
            m = hypercube_mat(dim=d, **kwargs, if_print=if_print)
        else:
            raise Exception("Invalid dimension {} for hypercube matrix".format(d))
    elif t == "OW":
        m = one_way_matrix(dim=d, **kwargs, if_print=if_print)
    elif t == "RAND":
        m = random_mat(dim=d, **kwargs, if_print=if_print)
        
    return m

## Calculating Attributes ------------------------------------------------

def calc_lyap_bremen(sim, CML): ## https://doi.org/10.1016/S0167-2789(96)00216-3
    matrix = CML.matrix
    assert len(matrix.shape) == 2
    assert matrix.shape[0] == matrix.shape[1]
    assert len(sim.shape) == 3
    assert matrix.shape[0] == sim.shape[1]
    assert sim.shape[2] == 1
    
    sim_len = sim.shape[0]
    dim = CML.dim
    
    jacobs = np.empty((sim_len, dim, dim))
    for i in range(sim_len):
        jacobs[i] = CML.jacobian(sim[i])
    
    lyap_spectrum = lyap_bremen(jacobs)

    lyap_spectrum[np.argsort(-lyap_spectrum)]
    
    return lyap_spectrum

def lyap_bremen(jacobs): ## https://doi.org/10.1016/S0167-2789(96)00216-3
    nsteps = jacobs.shape[0]
    dim = jacobs.shape[1]
    
    J = jacobs
    Q = np.eye(N=dim, M=dim)
    lyap_spectrum = np.zeros(dim)
    
    for i in range(nsteps):
        B = np.matmul(J[i], Q)
        Q, R = np.linalg.qr(B, mode='complete')
        lyap_spectrum += np.log(np.diag(np.abs(R)))
    
    lya = lyap_spectrum/nsteps
        
    return lya

def calc_ky(lya_spectrum):
    
    lya_desc = lya_spectrum[np.argsort(-lya_spectrum)]
    cum_lya = np.cumsum(lya_desc)

    if lya_desc[0] <= 0:
        # all lyapunovs negative
        warnings.warn("Warning: System is not chaotic")
        return None 
        
    top_dim = None
    for i in range(len(lya_desc)):
        if cum_lya[i] >= 0:
            top_dim = i
            
    if top_dim == None:
        print("Lyapunov spectrum: {}".format(lya_desc))
        print("Cumulative sum of Lyapunov spectrum: {}".format(cum_lya))
        print("Topological dimenesion: {}".format(top_dim))
        #raise Exception("Error computing topological dimension")
        return None

    if top_dim == len(lya_spectrum) - 1 or cum_lya[top_dim] == 0:
        ## if the cumulative sum never reaches 0
        ## or if the lyapunovs sum to exactly zero
        ## then ky_dim is system dim
        ky_dim = len(lya_spectrum)        
    else:
        ky_dim = (top_dim + 1) + (1/abs(lya_desc[top_dim + 1]))*(cum_lya[top_dim]) 
    ## (top_dim + 1) as indexing in python starts at 0
    return ky_dim

def check_periodicity(sim, return_period=True):
    unique_vals, unique_counts = np.unique(sim, return_counts=True, axis=0)
    periodic =  np.max(unique_counts) > 1

    if return_period:
        if periodic:
            period = len(unique_vals)
        else:
            period = None
        return periodic, period
    
    else:
        return periodic

def calc_dfa(series, if_plot=False):
    mean = np.mean(series)
    cusum = np.cumsum(series - mean)
    pydfa = fathon.DFA(cusum)
    max_wind = int(round(len(series)/4) + 1)
    wins = np.arange(4, max_wind, dtype=np.int64)
    
    n, F = pydfa.computeFlucVec(wins)
    H, H_intercept = pydfa.fitFlucVec()
    if if_plot:
        log_n = np.log(n)
        h_fit = np.exp(H*log_n+H_intercept)
        fig, ax = plt.subplots()
        ax.plot(n, F, marker='o')
        ax.plot(n, h_fit)
        ax.set_xscale('log', base=np.e)
        ax.set_yscale('log', base=np.e)
        plt.show()
    return H

def calc_LDDP(series, if_plot=False, npoints=50000):
    cdf = scipy.stats.ecdf(series)
    x = np.linspace(0,1, npoints)
    
    derivs = scipy.differentiate.derivative(cdf.cdf.evaluate, x).df
    if not np.all(derivs >= 0):
        derivs = np.where(derivs < 0, 0, derivs)
        
    num_err = scipy.integrate.simpson(derivs, dx = x[1]) - 1
    
    if if_plot:
        plt.figure()
        plt.plot(x, derivs)
        plt.show()
        plt.close()

    ## this raises divide/multiply warnings with zeros but doesn't use those results
    with np.errstate(divide='ignore', invalid='ignore'): 
        fn = np.where(derivs != 0, derivs * np.log(derivs), 0)

    h = scipy.integrate.simpson(fn, dx = x[1]) * -1

    return h, num_err

def calc_LDDP_corrected(series, if_plot=False, npoints=50000):
    cdf = scipy.stats.ecdf(series)
    x = np.linspace(0,1, npoints)
    
    derivs = scipy.differentiate.derivative(cdf.cdf.evaluate, x).df
    if not np.all(derivs >= 0):
        derivs = np.where(derivs < 0, 0, derivs)
        
    scaled_derivs = derivs/scipy.integrate.simpson(derivs, dx = x[1])
    num_err = scipy.integrate.simpson(scaled_derivs, dx = x[1]) - 1
    
    if if_plot:
        plt.figure()
        plt.plot(x, scaled_derivs)
        plt.show()
        plt.close()

    ## this raises divide/multiply warnings with zeros but doesn't use those results
    with np.errstate(divide='ignore', invalid='ignore'): 
        fn = np.where(scaled_derivs != 0, scaled_derivs * np.log(scaled_derivs), 0)

    h = scipy.integrate.simpson(fn, dx = x[1]) * -1

    return h, num_err

## CALC multiple attributes

def calc_system_metrics(sim, cml):
    start_time = time.time()
    dim = cml.dim
    lyap_spectrum = calc_lyap_bremen(sim, cml)
    largest_lyap = max(lyap_spectrum)
    n_pos_lyap = len([l for l in lyap_spectrum if l > 0])
    prop_pos_lyap = n_pos_lyap / dim
    chaotic = largest_lyap > 0
    hyperchaotic = n_pos_lyap >= 2
    ks_ent = (1/cml.dim)*np.sum(np.where(lyap_spectrum > 0, lyap_spectrum, 0))
    ky_dim = calc_ky(lyap_spectrum)
    periodic, period = check_periodicity(sim, return_period=True)
    metric_dict = {
        "dim": dim
        ,"lyapunov_spectrum": lyap_spectrum
        ,"chaotic": chaotic
        ,"hyperchaotic": hyperchaotic
        ,"n_pos_lyap": n_pos_lyap
        ,"prop_pos_lyap": prop_pos_lyap
        ,"ky_dim": ky_dim
        ,"ks_ent": ks_ent
        ,"system_periodic": periodic
        ,"system_period": period
    }
    for d in range(dim):
        metric_dict["lyap_{}".format(d)] = lyap_spectrum[d]
    end_time = time.time()
    metric_dict["system_metric_time"] = end_time - start_time
    return metric_dict

def calc_series_metrics(series, d):
    start_time = time.time()
    hurst_exp = nolds__hurst(series, corrected=True)

    if d == 1:
        embed_dim =1
    else:
        _, embed_dim = FNN_n(series, tau=1, maxDim=(2*d + 1), method='cao')


    embed_success = True

    if embed_dim == 0:
        embed_dim = 2*d + 1
        embed_success = False

    mpl.rcParams.update(mpl.rcParamsDefault) ## running fnn_n breaks matplotlib default params
    fisher, _ = fisher_information(series, delay=1, dimension=embed_dim)

    if embed_dim > 1:
        svd = ant__svd(series, order=embed_dim, delay=1, normalize=False)
    else:
        svd = None

    skewness = scipy.stats.skew(series)
    skew_test_p = scipy.stats.skewtest(series).pvalue
    kurt = scipy.stats.kurtosis(series)
    kurt_test_p = scipy.stats.kurtosistest(series).pvalue

    mean = np.mean(series)
    std = np.std(series)

    coeff_var = std/mean

    dfa = calc_dfa(series=series)

    diff_entropy = scipy.stats.differential_entropy(series)

    lddp, lddp_err = calc_LDDP(series=series)

    periodic, period = check_periodicity(series, return_period=True)
    
    metric_dict = {
        "hurst": hurst_exp
        ,"embed_dim": embed_dim
        ,"embed_success": embed_success
        ,"fisher": fisher
        ,"svd": svd
        ,"skewness": skewness
        ,"skew_test_p": skew_test_p
        ,"kurtosis": kurt
        ,"kurt_test_p": kurt_test_p
        ,"series_mean": mean
        ,"series_std": std
        ,"coefficient_variation": coeff_var
        ,"dfa": dfa
        ,"differential_entropy": diff_entropy
        ,"lddp": lddp
        ,"lddp_distribution_area_err": lddp_err
        ,"series_periodic": periodic
        ,"series_period": period
    }
    end_time = time.time()
    metric_dict["series_metric_time"] = end_time - start_time
    return metric_dict

def calc_point_metrics_knn(fitted_knn, test_in, test_targ, delta_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]):
    predictions = fitted_knn.predict(test_in)

    result_dict = {}
    abs_err = np.abs(predictions - test_targ)
    result_dict["max_abs_err"] = np.max(abs_err)
    result_dict["med_abs_err"] = np.median(abs_err)
    result_dict["mae"] = np.mean(abs_err)
    result_dict["mse"] = mse(test_targ, predictions)
    result_dict["rmse"] = rmse(test_targ, predictions)
    result_dict["mape"] = mape(test_targ, predictions)
    result_dict["rmsle"] = rmsle(test_targ, predictions)
    result_dict["R_squared"] = fitted_knn.score(test_in, test_targ)

    neigh_dists, neigh_inds = fitted_knn.kneighbors(test_in, n_neighbors=1, return_distance=True)
    neigh_dists = neigh_dists.flatten()
    
    result_dict["delta_min"] = np.min(neigh_dists)
    result_dict["delta_max"] = np.max(neigh_dists)
    result_dict["delta_mean"] = np.mean(neigh_dists)
    percentiles = np.percentile(neigh_dists, delta_percentiles)
    for i in range(len(percentiles)):
        result_dict["delta_p{}".format(delta_percentiles[i])] = percentiles[i]
    result_dict["delta_sd"] = np.std(neigh_dists)

    return result_dict

def calc_series_err_metrics(test_dts, pred_dts):
    result_dict = {}
    abs_err = np.abs(pred_dts - test_dts)
    result_dict["series_max_abs_err"] = np.max(abs_err)
    result_dict["series_med_abs_err"] = np.median(abs_err)
    result_dict["series_mae"] = np.mean(abs_err)
    result_dict["series_mse"] = mse(test_dts, pred_dts)
    result_dict["series_rmse"] = rmse(test_dts, pred_dts)
    result_dict["series_mape"] = mape(test_dts, pred_dts)
    result_dict["series_rmsle"] = rmsle(test_dts, pred_dts)
    return result_dict

def pred_test_dts_knn(test_dts, model, embed_dim, calc_deltas=True):
    n_series = test_dts.shape[0]
    test_len = test_dts.shape[1]
    test_pred = np.zeros(test_dts.shape)
    
    test_pred[:, 0:embed_dim] = test_dts[:, 0:embed_dim] ## for each test series the first embed_dim values are provided
    
    for i in range(embed_dim, test_len):
        inp = test_pred[:, i-embed_dim:i]     
        # print(inp.shape, model.predict(inp).shape, test_pred[:, i].shape) 
        test_pred[:, i] = np.reshape(model.predict(inp), (n_series,))

    if not calc_deltas:
        return test_pred
    else:
        initial_inps = test_pred[:, 0:embed_dim]
        initial_dists, _ = model.kneighbors(initial_inps, n_neighbors=1, return_distance=True)
        initial_dists = initial_dists.flatten()
        return test_pred, initial_dists
    
def calc_pred_h(test_dts, pred_dts, embed_dim, thresh=0.01, percentiles=[1, 5, 10, 25, 50, 75, 90, 95, 99], if_plot=False):
    
    diff_all = np.abs(test_dts - pred_dts) ##compute abosolute error
    assert np.all(diff_all[:, 0:embed_dim] == 0) ## the first embed_dim values in each series are provided so should have no error
    diff = diff_all[:, embed_dim:] ## remove the provided values
    diff_acc = np.maximum.accumulate(diff, axis=1)

    if if_plot:
        avg_diff = np.mean(diff_acc, axis=0)
        max_diff = np.max(diff_acc, axis=0)
        min_diff = np.min(diff_acc, axis=0)
        lower_quart = np.percentile(diff_acc, 25, axis=0)
        upper_quart = np.percentile(diff_acc, 75, axis=0)
        iters = np.arange(1, len(avg_diff) + 1)
        
        plt.figure()
        plt.title("Error After N Iterations")
        plt.plot(iters, avg_diff, label="AVG")
        plt.plot(iters, max_diff, label="MAX")
        plt.plot(iters, min_diff, label="MIN")
        plt.plot(iters, lower_quart, label="25%")
        plt.plot(iters, upper_quart, label="75%")
        plt.hlines(y=thresh, xmin=0, xmax=1.05*max(iters), ls='--', color='black')
        plt.xlim(0, 1.05*max(iters))
        plt.ylabel("Absolute Error")
        plt.xlabel("No. Iterations")
        plt.legend()
        plt.show()
        plt.close()
        
    n_acc_iters = np.argmax(diff_acc > thresh, axis=1).astype(float)

    # if np.any(diff_acc[:, -1] <= thresh): ## if the error on the final predicted value for any series is below threshold:
    #     raise Exception("Entire series predicted accurately")
    
    perfect_pred_c = 0

    for i in range(len(n_acc_iters)):
        if diff_acc[i, -1] <= thresh:
            n_acc_iters[i] = np.inf
            perfect_pred_c += 1
        else:
            idx = int(n_acc_iters[i])
            assert diff_acc[i, idx] > thresh
            if idx > 0:
                assert diff_acc[i, idx-1] <= thresh

    result_dict = {}

    result_dict["pred_h_min t={}".format(thresh)] = np.min(n_acc_iters)
    result_dict["pred_h_max t={}".format(thresh)] = np.max(n_acc_iters)
    result_dict["pred_h_mean t={}".format(thresh)] = np.mean(n_acc_iters)
    percentile_arr = np.percentile(n_acc_iters, percentiles)
    percentile_arr[np.isnan(percentile_arr)] = np.inf
    for i in range(len(percentile_arr)):
        result_dict["pred_h_p{} t={}".format(percentiles[i], thresh)] = percentile_arr[i]
    result_dict["pred_h_sd t={}".format(thresh)] = np.std(n_acc_iters)
    result_dict["pred_h_perfect_pred_count t={}".format(thresh)] = perfect_pred_c
    
            
    if if_plot:
        n_acc_iters_noinf = n_acc_iters[np.isfinite(n_acc_iters)]
        if len(n_acc_iters_noinf) == 0:
            return result_dict

        plt.figure()
        plt.title("Prediction Horizon; t={}".format(thresh))
        hist = plt.hist(n_acc_iters_noinf, 
                        np.arange(-0.5, n_acc_iters_noinf.max() + 1), 
                        edgecolor = "black")
        plt.vlines(x=result_dict["pred_h_mean t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='red',
                    label="Mean of number of accurate iters: " + str(round(result_dict["pred_h_mean t={}".format(thresh)], 4)))
        plt.vlines(x=result_dict["pred_h_p50 t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='green',
                    label="Median of number of accurate iters: " + str(round(result_dict["pred_h_p50 t={}".format(thresh)], 4)))
        plt.vlines(x=result_dict["pred_h_p25 t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='black', ls='--',
                    label="Lower quartile")
        plt.vlines(x=result_dict["pred_h_p75 t={}".format(thresh)], ymin=0, ymax=1.2*np.max(hist[0]), color='black', ls='--',
                    label="Upper quartile")
        plt.xlabel("Iterations Forward below " + str(thresh) + " error")
        plt.ylabel("Count")
        plt.ylim(0, 1.1*np.max(hist[0]))
        plt.xlim(n_acc_iters_noinf.min()-1, n_acc_iters_noinf.max()+1)
        plt.legend()
        plt.show()
        plt.close()

    return result_dict

def calc_knn_metrics(mod_data, testing_data, n_test_series, test_series_len, embed_dim, thresh_vals=[0.001, 0.01, 0.05],
               if_plot=False, percentile_vals=[1, 5, 10, 20, 25, 30, 40, 50, 60, 70, 75, 80, 90, 95, 99]):
    start_time = time.time()
    metric_dict = {}
    
    ## datasets
    n_train_points = len(mod_data) - embed_dim
    n_test_points = len(testing_data) - embed_dim
    ## training data
    train_in = np.empty((n_train_points, embed_dim))
    train_out = np.empty((n_train_points, 1))
    for i in range(n_train_points):
        train_in[i] = mod_data[i:i+embed_dim]
        train_out[i] = mod_data[i+embed_dim]
    ## testing data (points)
    test_in = np.empty((n_test_points, embed_dim))
    test_out = np.empty((n_test_points, 1))
    for i in range(n_test_points):
        test_in[i] = testing_data[i:i+embed_dim]
        test_out[i] = testing_data[i+embed_dim]
    ## testing data (series)
    test_series = np.array(np.split(testing_data, n_test_series))
    assert np.all(test_series.shape == (n_test_series, test_series_len))

    ## knn model
    knn_model = KNeighborsRegressor(n_neighbors=1)
    knn_model.fit(train_in, train_out)

    ## calc_metrics
    metric_dict.update(calc_point_metrics_knn(fitted_knn=knn_model, test_in=test_in, test_targ=test_out,
                                              delta_percentiles = percentile_vals))
    
    pred_dts, init_dists = pred_test_dts_knn(test_dts=test_series, model=knn_model, embed_dim=embed_dim, calc_deltas=True)
    metric_dict["series_initial_delta_min"] = np.min(init_dists)
    metric_dict["series_initial_delta_max"] = np.max(init_dists)
    metric_dict["series_initial_delta_mean"] = np.mean(init_dists)
    percentiles = np.percentile(init_dists, percentile_vals)
    for i in range(len(percentiles)):
        metric_dict["series_initial_delta_p{}".format(percentile_vals[i])] = percentiles[i]
    metric_dict["series_initial_delta_sd"] = np.std(init_dists)

    metric_dict.update(calc_series_err_metrics(test_series, pred_dts))
    
    for t in thresh_vals:
        metric_dict.update(calc_pred_h(test_dts=test_series, pred_dts=pred_dts, embed_dim=embed_dim, thresh=t,
                                       percentiles=percentile_vals, if_plot=if_plot))
    
    dict_names = list(metric_dict.keys())
    dict_vals = list(metric_dict.values())
    new_names = ["knn_" + k for k in dict_names]
    final_dict = dict(zip(new_names, dict_vals))
        
    end_time = time.time()
    final_dict["knn_metric_time"] = end_time - start_time
    
    return final_dict

def sim_system(sys_dict):
    start_time = time.time()
    result_dict = sys_dict.copy()
    
    sim_start = time.time()
    sim = np.empty((sys_dict["total_sim_len"], sys_dict["dim"], 1))
    sim[0] = sys_dict["sv"].reshape(sim[0].shape)
    origin_hitcount = 0
    for n in range(1, sys_dict["total_sim_len"]):
        sim[n] = sys_dict["cml"].step(sim[n-1])
        if np.all(sim[n] == 0):
            sim[n] = np.zeros(d, 1) + (np.random.rand() * (10**(-6))) ## if system hits [0, ..., 0] by chance, give small push to see if is attractive 
            origin_hitcount += 1
    sim_discard = sim[sys_dict["discard_len"]:]
    
    sim_1d = sim_discard[:, sys_dict["analysis_dim"], 0]
    
    np.save(sys_dict["sim_file"], sim_1d)
    result_dict["origin_hits"] = origin_hitcount
    result_dict["sim_shape"] = sim.shape
    result_dict["sim_discard_shape"] = sim_discard.shape
    result_dict["sim_1d_shape"] = sim_1d.shape
    sim_end = time.time()
    result_dict["sim_time"] = sim_end-sim_start

    system_metrics = calc_system_metrics(sim=sim_discard, cml=sys_dict["cml"])
    result_dict.update(system_metrics)

    if not result_dict["chaotic"] and result_dict["system_periodic"]:
        return result_dict

    series_metrics = calc_series_metrics(series=sim_1d, d=sys_dict["dim"])
    result_dict.update(series_metrics)

    mod_data, testing_data = np.split(sim_1d, [sys_dict["modelling_len"]])
    assert len(mod_data) == sys_dict["modelling_len"]
    assert len(testing_data) == sys_dict["testing_len"]
    knn_metrics = calc_knn_metrics(mod_data=mod_data, testing_data=testing_data,
                                      n_test_series=result_dict["n_tests"], test_series_len=result_dict["test_len"],
                                      embed_dim=result_dict["embed_dim"], if_plot=False)
    result_dict.update(knn_metrics)

    end_time = time.time()
    result_dict["full_system_time"] = end_time-start_time
 
    return result_dict

def test_import():
    return True

def sim_strange_system(sys_dict):
    start_time = time.time()
    result_dict = sys_dict.copy()
    
    sim_start = time.time()
    sim = np.empty((sys_dict["total_sim_len"], sys_dict["dim"], 1))
    sim[0] = sys_dict["sv"].reshape(sim[0].shape)
    origin_hitcount = 0
    for n in range(1, sys_dict["total_sim_len"]):
        sim[n] = sys_dict["cml"].step(sim[n-1])
        if np.all(sim[n] == 0):
            sim[j] = np.zeros(d, 1) + (np.random.rand() * (10**(-6))) ## if system hits [0, ..., 0] by chance, give small push to see if is attractive 
            origin_hitcount += 1
    sim_discard = sim[sys_dict["discard_len"]:]

    if sys_dict["run_type"] == "original":
        sim_1d = sim_discard[:, sys_dict["analysis_dim"], 0]
        sim_file = "C:/Users/B00955739/OneDrive - Ulster University/Documents/PhD/Results/Predicting_predictability/" + "Sims/main_run_v2/system_{}.npy".format(sys_dict["system_id"])
        prev_sim = np.load(sim_file)
        result_dict["sim_reproduced"] = np.all(sim_1d == prev_sim)
    
    result_dict["origin_hits"] = origin_hitcount
    # result_dict["sim_1d"] = sim_1d
    # result_dict["sim_discard"] = sim_discard
    result_dict["sim_discard_shape"] = sim_discard.shape
    sim_end = time.time()
    result_dict["sim_time"] = sim_end-sim_start
    
    lyap_spectrum = calc_lyap_bremen(sim_discard, sys_dict["cml"])
    
    result_dict["largest_lyap"] = max(lyap_spectrum)
    result_dict["lyap_spectrum"] = lyap_spectrum 

    period_dict = calc_period(sim_discard)
    result_dict.update(period_dict)

    # unique_vals, unique_counts = np.unique(sim_discard, return_counts=True, axis=0)
    # result_dict["periodic_precise"] = np.any(unique_counts > 1)
    # result_dict["estimated_period_precise"] = len(unique_vals)
    # result_dict["unique_vals_precise"] = unique_vals
    # result_dict["unique_counts_precise"] = unique_counts
    # if np.any(unique_counts > 1):
    #     result_dict["exact_period_precise"], result_dict["exact_period_precise_success"] = get_exact_period(sim_discard)

    # rounded_sim = np.round(sim_discard, 8)
    # unique_vals, unique_counts = np.unique(rounded_sim, return_counts=True, axis=0)
    # result_dict["periodic_rounded"] = np.any(unique_counts > 1)
    # result_dict["estimated_period_rounded"] = len(unique_vals)
    # result_dict["unique_vals_rounded"] = unique_vals
    # result_dict["unique_counts_rounded"] = unique_counts
    # if np.any(unique_counts > 1):
    #     result_dict["exact_period_rounded"], result_dict["exact_period_rounded_success"] = get_exact_period(rounded_sim)
    
    end_time = time.time()
    result_dict["full_system_time"] = end_time-start_time
 
    return result_dict

def get_exact_period(sim):
    final_val = sim[-1]
    period = 0

    for test_period in range(1, len(sim)):
        check_val = sim[-(test_period+1)]
        if np.all(check_val == final_val):
            period = test_period
            break

    # while test_period < len(sim) and not np.all(check_val == final_val):
    #     test_period += 1
    #     check_val = sim[-(test_period+1)]
    
    success = False
    if period + 2 <= len(sim) and period > 0:
        if np.all(sim[-1] == sim[-(period+1)]) and np.all(sim[-2] == sim[-(period+2)]):
            success = True

    return period, success

def calc_period(sim):
    unique_vals, unique_counts = np.unique(sim, return_counts=True, axis=0)

    if np.any(unique_counts > 1):
        periodic = True
    else:
        periodic = False

    n_unique_vals = len(unique_vals)

    if periodic:
        exact_period, exact_period_success = get_exact_period(sim)
    else:
        exact_period = np.inf
        exact_period_success = True

    if periodic:
        rounded_unique_vals = np.unique(np.round(sim[-int(exact_period + 3):], 6), axis=0)
        rounded_period = len(rounded_unique_vals)

    else:
        rounded_period = np.inf
        rounded_unique_vals = np.zeros(sim[0].shape)

    period_dict = {
        "periodic": periodic,
        "n_unique_vals": n_unique_vals,
        "exact_period": exact_period,
        "exact_period_success": exact_period_success,
        "rounded_period": rounded_period,
        "rounded_period_vals": rounded_unique_vals
    }
    return period_dict
