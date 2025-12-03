## New tools, developed on purpose for this repository
## Some of them are used in `simpleMD.py` to extend its functionalities to the refinement on-the-fly.

import numpy as np
import numba

@numba.njit(cache=True, fastmath=True)
def _compute_rgyr2(positions: np.ndarray, cell: np.ndarray = None, is_forces: bool = False):
    """
    Compute the squared radius of gyration of a set of positions.
    This quantity is computed as the mean squared distance of particles from their geometric center.
    If `cell` is provided, distances are computed using minimum image convention for periodic boundaries.

    Parameters
    ----------
    positions: np.ndarray
        Array of shape (N, 3) containing the Cartesian coordinates of N particles.
    
    cell: array-like or None
        Optional box dimensions for periodic boundary conditions.
        If provided, `cell` should be an array/list of length 3 [Lx, Ly, Lz].
        If None, no periodic boundary correction is applied.
    
    is_forces: Bool
        Boolean variable, if True then compute the derivatives of the squared gyration radius with respect to the
        atomic coordinates `positions`.

    Returns
    -------
    float
        The squared radius of gyration (Rg^2) of the positions.
    """
    if is_forces: forces = np.zeros(shape=positions.shape)

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

        if cell is not None:
            distancex -= np.floor(distancex/cell[0] + 0.5)*cell[0]
            distancey -= np.floor(distancey/cell[1] + 0.5)*cell[1]
            distancez -= np.floor(distancez/cell[2] + 0.5)*cell[2]
        
        rgyr2 += distancex**2 + distancey**2 + distancez**2

        if is_forces:
            forces[i, 0] -= distancex
            forces[i, 1] -= distancey
            forces[i, 2] -= distancez

    if not is_forces: forces = None
    else: forces = forces*2/len(positions)
    return rgyr2, forces

@numba.njit(cache=True, fastmath=True)
def _compute_SAXS(positions: np.ndarray, cell: np.ndarray = None, q_SAXS: np.ndarray = 0.1*(np.arange(10) + 1),
                  a_SAXS: np.ndarray = np.zeros(4), b_SAXS: np.ndarray = np.zeros(4), c_SAXS: float = 1.0,
                  is_forces: bool = False):
    """ 
    Low-level backend that employs `numba`. See `compute_SAXS` for documentation.
    Additional parameter: `is_forces` (if True, return also forces, None otherwise).
    """

    saxs = np.zeros(len(q_SAXS))
    if is_forces:
        forces = []
        for q in range(len(q_SAXS)):
            forces.append(np.zeros(shape=positions.shape))

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
                if cell is not None:
                    distancex -= np.floor(distancex/cell[0] + 0.5)*cell[0]
                    distancey -= np.floor(distancey/cell[1] + 0.5)*cell[1]
                    distancez -= np.floor(distancez/cell[2] + 0.5)*cell[2]

                distance2 = distancex**2 + distancey**2 + distancez**2
                distance = np.sqrt(distance2)

                saxs[q] += np.sin(q_SAXS[q]*distance)/(q_SAXS[q]*distance)

                if is_forces:
                    fmod = -(q_SAXS[q]*distance*np.cos(q_SAXS[q]*distance) - np.sin(q_SAXS[q]*distance))/(q_SAXS[q]*distance**3)
                    fx = fmod*distancex
                    fy = fmod*distancey
                    fz = fmod*distancez
                    forces[q][i, 0] += fx
                    forces[q][i, 1] += fy
                    forces[q][i, 2] += fz
                    forces[q][j, 0] -= fx
                    forces[q][j, 1] -= fy
                    forces[q][j, 2] -= fz

    saxs = saxs*scattering2

    if not is_forces: forces = None
    else:
        for q in range(len(forces)): forces[q] = forces[q]*scattering2
    return saxs, forces  # numba requires same output (no two different outputs depending on is_force)

def compute_SAXS(positions: np.ndarray, cell: np.ndarray = None, q_SAXS: np.ndarray = 0.1*(np.arange(10) + 1),
                 a_SAXS: np.ndarray = np.zeros(4), b_SAXS: np.ndarray = np.zeros(4), c_SAXS: float = 1.0):
    """
    Compute the SAXS spectrum through the Debye equation.
    
    Parameters
    ----------
    positions: np.ndarray
        Array of shape (N, 3) containing the Cartesian coordinates of N particles.
    
    cell: array-like or None
        Optional box dimensions for periodic boundary conditions.
        If provided, `cell` should be an array/list of length 3 [Lx, Ly, Lz].
        If None, no periodic boundary correction is applied.

    q_SAXS: array-like
        Array of values for the magnitude of the scattering vectors.

    a_SAXS, b_SAXS: array-like
        Arrays of values required to compute the scattering factor, together with `c_SAXS`.
    
    c_SAXS: float
        Together with `a_SAXS` and `b_SAXS`, these variables are used to compute the scattering factor (or form factor).
        By default, it is `scattering = 1`.

    Returns
    -------
    np.ndarray
        The array of values with the SAXS spectrum.

    Notes
    -----
    This is a wrapper for `_compute_SAXS` that returns SAXS spectrum only (not forces), in this way I do not need to
    import numba in the notebook to run `compute_SAXS`, clearly numba has to be imported anyway in the module.
    """
    saxs = _compute_SAXS(positions, cell, q_SAXS, a_SAXS, b_SAXS, c_SAXS, False)[0]
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

def compute_dkl(p, p0, if_zero=False):
    """
    Compute the Kullback-Leibler divergence between `p` and `p0`.
    If `if_zero` is True, then remove from `p` and `p0` the points with `p0 = 0` so that no `inf` value
    will be returned. To check that this modification is just due to statistical fluctuation, return also
    the total removed probability.
    """
    
    p0 = p0[p != 0]
    p = p[p != 0]

    if if_zero:
        tot = np.sum(p[p0 == 0])
        p = p[p0 != 0]
        p0 = p0[p0 != 0]
    
    dkl = np.sum(p*np.log(p/p0))
    
    if if_zero: return dkl, tot
    else: return dkl

