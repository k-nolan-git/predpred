import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools as it
import pandas as pd
from teaspoon.parameter_selection.FNN_n import FNN_n
import time
import warnings
import torch
import torch.utils.data as data_utils
from torch import nn
import torch.optim as optim
import sklearn as skl
import copy

## Ensure import has worked ----------------------------------------------

def test_import():
    return True

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

## Simulate system ------------------------------------------------

def simulate_system(_cml, _sv, _nsteps):
    origin_hits = 0
    
    dim = _cml.dim
    sim = np.empty((_nsteps, dim, 1))
    sim[0] = _sv.reshape(sim[0].shape)
    
    for i in range(1, _nsteps):
        sim[i] = _cml.step(sim[i-1])
        if np.all(sim[i] == 0):
            sim[i] = sim[0] + (np.random.rand() * 10**(-6))
            origin_hits += 1
            
    return sim, origin_hits

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

def calc_embed_dim(series, dim, return_dict=False):
    embed_sucess = True
    if dim == 1:
        return 1, embed_sucess
    else:
        maxDim = (2*dim) + 2
        _, cao_embed_dim = FNN_n(ts=series, tau=1, maxDim=maxDim, method='cao')
        _, str_embed_dim = FNN_n(ts=series, tau=1, maxDim=maxDim, Rtol=10, method='strand')
        _, def_embed_dim = FNN_n(ts=series, tau=1, maxDim=maxDim)
        embed_dims = [cao_embed_dim, str_embed_dim, def_embed_dim]
        embed_successes = [x != 0 for x in embed_dims]
        embed_dim = np.max(embed_dims)
        if embed_dim == 0:
            embed_dim = (2*dim) + 1
            embed_sucess = False

        mpl.rcParams.update(mpl.rcParamsDefault) ## FNN_n breaks matplotlib
        if return_dict:
            info_dict = {
                "cao": cao_embed_dim, "cao_success": embed_successes[0],
                "strand": str_embed_dim, "strand_success": embed_successes[1],
                "default": def_embed_dim, "default_success": embed_successes[2],
                }
            return embed_dim, embed_sucess, info_dict
        else:
            return embed_dim, embed_sucess
    
## ML Functions ------------------------------------------------------------

def create_dataset(series, embed_dim, lf): 
    ## 1d series -> Tuple of (inputs, outputs); input (series_len-embed_dim-lf+1, embed_dim, 1) and outputs (series_len-embed_dim-lf+1, lf, 1)
    s_len = len(series)
    data_size = s_len-embed_dim-lf+1
    inputs = np.empty((data_size, embed_dim, 1))
    outputs = np.empty((data_size, lf, 1)) 
    
    for i in range(data_size):
        inputs[i] = series[i:i+embed_dim].reshape(embed_dim, 1)
        outputs[i] = series[i+embed_dim:i+embed_dim+lf].reshape(lf, 1)
        
    assert inputs.flatten()[0] == series[0]
    assert outputs.flatten()[-1] == series[-1]
    
    return (inputs, outputs)
    
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