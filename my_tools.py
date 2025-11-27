## New tools, developed on purpose for this repository

import numpy as np
import numba

@numba.njit(cache=True, fastmath=True)
def _compute_rgyr2(positions, cell):
    meanx = 0.0
    meany = 0.0
    meanz = 0.0
    
    for i in range(len(positions)):
        meanx += positions[i, 0]
        meany += positions[i, 1]
        meanz += positions[i, 2]
    
    meanx /= len(positions)
    meany /= len(positions)
    meanz /= len(positions)
    
    rgyr2 = 0.0

    for i in range(len(positions)):
        distancex = positions[i, 0] - meanx
        distancey = positions[i, 1] - meany
        distancez = positions[i, 2] - meanz
        distancex -= np.floor(distancex/cell[0] + 0.5)*cell[0]
        distancey -= np.floor(distancey/cell[1] + 0.5)*cell[1]
        distancez -= np.floor(distancez/cell[2] + 0.5)*cell[2]
        rgyr2 += distancex**2 + distancey**2 + distancez**2
    
    rgyr2 /= len(positions)
    
    return rgyr2

@numba.njit(cache=True, fastmath=True)
def _compute_SAXS(positions, cell, q_SAXS, a_SAXS, b_SAXS, c_SAXS):

    saxs = np.zeros(len(q_SAXS))

    for q in range(len(q_SAXS)):
        q_factor = (q_SAXS[q]/(4*np.pi))**2
        scattering = c_SAXS
        for s in range(4):
            scattering += a_SAXS[s]*np.exp(-b_SAXS[s]*q_factor)

        scattering2 = scattering**2
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                distancex = positions[i, 0] - positions[j, 0]
                distancey = positions[i, 1] - positions[j, 1]
                distancez = positions[i, 2] - positions[j, 2]
                distancex -= np.floor(distancex/cell[0] + 0.5)*cell[0]
                distancey -= np.floor(distancey/cell[1] + 0.5)*cell[1]
                distancez -= np.floor(distancez/cell[2] + 0.5)*cell[2]

                distance2 = distancex**2 + distancey**2 + distancez**2
                distance = np.sqrt(distance2)
                saxs[q] += np.sin(q_SAXS[q]*distance)/(q_SAXS[q]*distance)

        saxs[q] = saxs[q]*scattering2

    return saxs

@numba.njit(cache=True, fastmath=True)
def gamma(x, s, s_exp):  # use capital letters only for Classes
    weights = np.exp(-x*(s - s_exp))
    gamma_val = np.log(np.sum(weights) / len(s))
    return gamma_val


def stable_softmax(logw):
    # logw: array (N,) di log-pesi
    m = np.max(logw)
    w = np.exp(logw - m)  # ora sicuro numericamente
    w /= np.sum(w)
    return w

def weighted_mean_var(values, logw):
    p = stable_softmax(logw)
    mu = np.sum(p * values)
    var = np.sum(p * (values - mu)**2)
    return mu, var, p

def compare_dicts(dict1, dict2):
    """ WARNING: it works only when attribute values are "single variables" (integer, floats) or np.ndarray,
    not more complicated data types """

    diff1 = set(dict1.keys()) - set(dict2.keys())
    diff2 = set(dict2.keys()) - set(dict1.keys())
    common = set(dict1.keys()) - diff1

    if diff1 == set(): diff1 = 'empty'
    if diff2 == set(): diff2 = 'empty'

    print('in dict1 but not in dict2: ', diff1)
    print('in dict2 but not in dict1: ', diff2)
    print('\ncommon attributes: ', list(common))

    print('\n\ndifferent values of common attributes: ')

    different_values = {}

    b = 0

    for k in list(common):
        if isinstance(dict1[k], np.ndarray) and isinstance(dict2[k], np.ndarray):
            if not np.array_equal(dict1[k], dict2[k]):
                different_values[k] = (dict1[k], dict2[k])
                print(k, different_values[k])
                b = 1
        else:
            if dict1[k] != dict2[k]:
                different_values[k] = (dict1[k], dict2[k])
                print(k, different_values[k])
                b = 1

    if b == 0: print('all the common attributes have equal values')

    return

