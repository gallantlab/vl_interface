import numpy as np
import cortex
import cortex.polyutils
from scipy.stats import linregress
import sys

from vl_interface import CC_INTERFACE

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)

def geodesic_path(a, b, surf, max_len=500, **kwargs):
    """Draws a path from point a to point b by picking the point that is a neighbor of a that is 
    closest to b, and then repeating that process until the path leads all the way to point b
    """
    path = [a]
    d = surf.geodesic_distance([b], **kwargs)
    while path[-1] != b:
        n = [v for v in surf.graph.neighbors(path[-1])]
        path.append(n[d[n].argmin()])
        if len(path) > max_len:
            return None
    return path

def anti_geodesic_path(a, b, border_pts, surf, max_len=500, **kwargs):
    """Draws a path from point a AWAY from point b, continues until it hits another of the border_pts
    """
    path = [a]
    d = surf.geodesic_distance([b], **kwargs)
    # Continue while last point of path is within the set of pts
    while path[-1] not in border_pts:
        n = [v for v in surf.graph.neighbors(path[-1])]
        #print(n)
        # Ran into an error where sometimes the path would just go back and forth between
        # two vertices, so this accounts for that error by not adding a vertex to the path
        # if it is already in the path, and adding the next best element instead
        while n[d[n].argmax()] in path:
            print("!!! USED REMOVAL LOOP !!!  Vertex: %d" % (n[d[n].argmax()]))
            n.remove(n[d[n].argmax()])
            if len(n)==0:
                return None
        path.append(n[d[n].argmax()])
        if len(path) > max_len:
            return None
    return path

def equidistant_verts(subject, spacing=2.5, max_verts=30000):
    """Better spacing and faster than tissots_subset
    """
    vert_arrays = []
    allcenters = []
    for num,hem in enumerate(["lh","rh"]):
        fidpts, fidpolys = cortex.db.get_surf(subject, "fiducial", hem)
        surf = cortex.polyutils.Surface(fidpts, fidpolys)
        n_vert = fidpts.shape[0]
        vert_array = np.zeros(n_vert)
        centers = [np.random.randint(n_vert)]
        vert_array[centers[0]] = 1

        while True:
            dists = surf.geodesic_distance(centers)
            possverts = np.nonzero(dists > spacing)[0]
            if not len(possverts):
                break
            if len(centers) >= max_verts:
                print("Reached limit of {} verts... exiting".format(max_verts))
                break
            # Pick the vertex that is closest to the existing verts 
            # While still covering another good chunk of cortex
            centervert = np.abs(dists - 2*spacing).argmin()
            centers.append(centervert)
            vert_array[centervert] = 1
            if len(centers)%500==0:
                print("Total verts is now {}...".format(len(centers)))
        
        vert_arrays.append(vert_array)
        allcenters.append(centers)
        print("Finished {}".format(hem))

    return vert_arrays, allcenters

def load_model_info(subject, xfm=None, model_type='tikreg', verbose=True, **kwargs):
    """Load movie and story model weights from specified locations
    By default, from the separate movie and listening experiments
    model_type can also be "reading", "banded", "shortfilms", "tikreg"
    optional kwargs are story_model and movie_model

    If xfm is provided, weights are projected into Vertex space and returned
    """
    sub = subject[:2]

    if verbose:
        print("Loading story and movie weights from: {}".format(model_type))
    if model_type='tikreg':
        swt, mwt = load_tikreg_weights(sub, **kwargs)
    else:
        raise NotImplementedError

    if xfm is not None:
        print("xfm provided, transforming into Vertex space...")
        mapper = cortex.get_mapper(subject, xfm, "line_nearest", recache=False)
        surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
        numl = surfs[0].pts.shape[0]
        # Get story data
        storyvol_wt = (swt, subject, xfm)
        storyverts_wt = mapper(storyvol_wt)
        storydata_wt = [storyverts_wt.data[:,:numl], storyverts_wt.data[:,numl:]]
        # Get movie data
        movievol_wt = (mwt, subject, xfm)
        movieverts_wt = mapper(movievol_wt)
        moviedata_wt = [movieverts_wt.data[:,:numl], movieverts_wt.data[:,numl:]]
        return storydata_wt, moviedata_wt
    else:
        return swt, mwt

def load_tikreg_weights(sub, story_model='story_f1', movie_model='movie_f1'):
    movie_file = "models/{subject}_banded_ridge_{model}_eng1000.ccd"
    story_file = "models/{subject}_banded_ridge_{model}_eng1000.ccd"
    mwt = cci.download_raw_array(movie_file.format(subject=sub, model=movie_model))
    swt = cci.download_raw_array(story_file.format(subject=sub, model=story_model))
    return swt, mwt

def load_pcs(pc_model='fb1'):
    pc_file = "models/listening_pcs_{model}_eng1000.ccd"
    grpc = cci.download_raw_array(pc_file.format(model=pc_model))
    return grpc

def scale_rgb_for_plotting(rgb):
    '''rgb : array_like, (3, nvox)
    Scales by standard deviation, clips at 3 SD
    Then sets that range to be between 0-255
    '''
    zrgb = rgb/np.std(rgb)
    clip_lim = 3
    crgb = np.clip(zrgb, -clip_lim, clip_lim)
    srgb = crgb/clip_lim/2.0 + 0.5
    rgb_map = (srgb * 255).astype(np.uint8)
    return rgb_map

def delay_signal(data, delays=[0, 1, 2, 3], fill=0):
    """
    >>> x = np.arange(6).reshape(2,3).T + 1
    >>> x
    array([[1, 4],
           [2, 5],
           [3, 6]])
    >>> delay_signal(x, [-1,2,1,0], fill=0)
    array([[2, 5, 0, 0, 0, 0, 1, 4],
           [3, 6, 0, 0, 1, 4, 2, 5],
           [0, 0, 1, 4, 2, 5, 3, 6]])
    >>> delay_signal(x, [-1,2,1,0], fill=np.nan)
    array([[  2.,   5.,  nan,  nan,  nan,  nan,   1.,   4.],
           [  3.,   6.,  nan,  nan,   1.,   4.,   2.,   5.],
           [ nan,  nan,   1.,   4.,   2.,   5.,   3.,   6.]])
    """
    if data.ndim == 1:
        data = data[...,None]
    n, p = data.shape
    out = np.ones((n, p*len(delays)), dtype=data.dtype)*fill

    for ddx, num in enumerate(delays):
        beg, end = ddx*p, (ddx+1)*p
        if num == 0:
            out[:, beg:end] = data
        elif num > 0:
            out[num:, beg:end] = data[:-num]
        elif num < 0:
            out[:num, beg:end] = data[abs(num):]
    return out

def unstack_wts(wts, ndelays, model_dims):
    '''Unstack a matrix of voxel weights based on the dimensions of each model,
    assuming that each dimension is in the order of model_dims

    Returns a list of matrices that are each features and all delays for that
    model
    '''
    model_nums = np.tile(np.repeat(range(len(model_dims)), model_dims),
                         ndelays)
    unstacked = [wts[model_nums==m, :] for m in range(len(model_dims))]
    return unstacked

def fdr_correct(pval, thres):
    """Find the fdr corrected p-value thresholds
    pval - vector of p-values
    thres - FDR level
    pID - p-value thres based on independence or positive dependence
    pN - Nonparametric p-val thres"""
    # remove NaNs
    p = pval[np.nonzero(np.isnan(pval)==False)[0]]
    p = np.sort(p)
    V = np.float(len(p))
    I = np.arange(V) + 1

    cVID = 1
    cVN = (1/I).sum()

    th1 = np.nonzero(p <= I/V*thres/cVID)[0]
    th2 = np.nonzero(p <= I/V*thres/cVN)[0]
    if len(th1)>0:
        pID = p[th1.max()]
    else:
        pID = -np.inf
    if len(th2)>0:
        pN =  p[th2.max()]
    else:
        pN = -np.inf

    return pID, pN


## Z-score -- z-score each column
zscore = lambda v: (v-v.mean(0))/v.std(0)
zscore.__doc__ = """Z-scores (standardizes) each column of [v]."""
zs = zscore

## Matrix corr -- find correlation between each column of c1 and the corresponding column of c2
mcorr = lambda c1,c2: (zs(c1)*zs(c2)).mean(0)
mcorr.__doc__ = """Matrix correlation. Find the correlation between each column of [c1] and the corresponding column of [c2]."""
