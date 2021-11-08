import numpy as np
from copy import copy
import cortex
from scipy.stats import linregress
from scipy.spatial.distance import cosine

from vl_interface import CC_INTERFACE
from vl_interface.line_io import load_lines, recon_line
from vl_interface.line_objects import BrainLine
from vl_interface.lobe_border import load_band
from vl_interface.utils import load_model_info, fdr_correct

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)


def spatial_perm_v_or_l(subject, all_lines, hem, line_num, v_or_l, orig_slope_diff,
                                storydata_wt, moviedata_wt):
    total_lines = len(all_lines[0]) + len(all_lines[1])

    line = all_lines[hem][line_num]
    current_n = len(line.subdiv_verts)
    current_verts = np.array(line.subdiv_verts)
    
    # Find the average vector for current line first
    # And then multiply by entire brain for other modality
    if v_or_l=='v':
        subset_moviedata_wt = moviedata_wt[line.hem][:,current_verts]
        avg_mwts = np.mean(subset_moviedata_wt, axis=1)
        sem_vector = avg_mwts / np.linalg.norm(avg_mwts, 2)
        current_ROI = np.dot(sem_vector, subset_moviedata_wt)
        all_other_proj = [np.dot(sem_vector, wt) for wt in storydata_wt]
    elif v_or_l=='l':
        subset_storydata_wt = storydata_wt[line.hem][:,current_verts]
        avg_swts = np.mean(subset_storydata_wt, axis=1)
        sem_vector = avg_swts / np.linalg.norm(avg_swts, 2)
        current_ROI = np.dot(sem_vector, subset_storydata_wt)
        all_other_proj = [np.dot(sem_vector, wt) for wt in moviedata_wt]
    
    shuf_proj = np.zeros((total_lines, current_n))
    shuf_ap = np.zeros((total_lines, current_n))
    count = 0
    for h in range(2):
        for shuf_line in all_lines[h]:
            shuf_n = len(shuf_line.subdiv_verts)
            # Sample with replacement b/c line subdiv might be smaller than original
            shuf_idx = np.random.choice(shuf_n, current_n)
            shuf_ap[count] = np.array(shuf_line.subdiv_ap)[shuf_idx]
            shuf_verts = np.array(shuf_line.subdiv_verts)[shuf_idx]
            shuf_proj[count] = all_other_proj[h][shuf_verts]
            count += 1
    
    X_current = np.array(line.subdiv_ap)
    X_all_current = np.vstack((X_current, np.ones_like(X_current)))
    XTX_current = np.dot(X_all_current, X_all_current.T)
    XTY_current = np.dot(X_all_current, current_ROI.T)
    current_out = np.dot(np.linalg.inv(XTX_current), XTY_current)
    
    X_all_shuf = np.stack((shuf_ap, np.ones_like(shuf_ap)))
    XTX_shuf = np.matmul(np.transpose(X_all_shuf, (1,0,2)), np.transpose(X_all_shuf, (1,2,0)))
    XTY_shuf = np.matmul(np.transpose(X_all_shuf, (1,0,2)), shuf_proj[:,:,np.newaxis])
    shuf_out = np.squeeze(np.matmul(np.linalg.inv(XTX_shuf), XTY_shuf))
    
    slope_current = current_out[0]
    int_current = current_out[1]
    slopes_shuf = shuf_out[:,0]
    ints_shuf = shuf_out[:,1]
    
    if v_or_l=='v':
        divs = np.array([slope_current/slopes_shuf, slopes_shuf/slope_current])
        idx = np.argmin(np.abs(divs), axis=0)
        div_metric = np.array([divs[idx[d],d] for d in range(divs.shape[1])])
        # Did some basic algebra to find this
        ap_split = (ints_shuf - int_current) / (slope_current - slopes_shuf)
    elif v_or_l=='l':
        divs = np.array([slopes_shuf/slope_current, slope_current/slopes_shuf])
        idx = np.argmin(np.abs(divs), axis=0)
        div_metric = np.array([divs[idx[d],d] for d in range(divs.shape[1])])
        # Did some basic algebra to find this
        ap_split = (int_current - ints_shuf) / (slopes_shuf - slope_current)
        
    slope_mag = np.mean([np.abs(slopes_shuf), np.ones_like(slopes_shuf) * np.abs(slope_current)], axis=0)
    cross_indicator = (np.logical_and(max(line.subdiv_ap) > ap_split,
                                      min(line.subdiv_ap) < ap_split)).astype(int)
    pos_indicator = (shuf_proj.mean(1) * current_ROI.mean() > 0).astype(int)
    
    perm_slope_diffs = -1*div_metric*slope_mag*cross_indicator*pos_indicator
    p_val = np.sum(perm_slope_diffs >= orig_slope_diff).astype(float) / len(perm_slope_diffs)

    return p_val

def gather_spatial_perm_v_or_l(subject, param_set_name, occ_distance=50):
    done = spatial_perm_v_or_l_finished(subject, param_set_name)
    if done:
        filename_centers = "line_data/{}_{}_h{}.hf5/center_verts"
        centers = [cci.download_raw_array(filename_centers.format(subject, param_set_name, h))
                   for h in range(2)]
        num_left = centers[0].shape[0]
        surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
        numl = surfs[0].pts.shape[0]
        bands = [load_band(subject, h, distance=occ_distance) for h in range(2)]
         
        # Have to do these loops to avoid MemoryErrors
        filename_indiv = "spatial_perm_v_or_l_pval/{}_{}_{}_{}.ccd"
        p_val = np.hstack([np.array([cci.download_raw_array(filename_indiv.format(subject, param_set_name, h, vert)).max()
                                     for vert in range(len(centers[h]))]) for h in range(2)])
        
        all_centers = np.array([vert+h*numl for h in range(2) for vert in centers[h]])
        occ_centers = np.array([vert+h*numl for h in range(2) for vert in centers[h] if vert in bands[h]])

        occ_idx = [c in occ_centers for c in all_centers]
        occ_p_val = p_val[occ_idx]
                
        p_val_file = "spatial_perm_v_or_l_pval/{}_{}.hf5"
        outdict = dict(p_val=p_val, centers=all_centers)
        occ_outdict = dict(p_val=occ_p_val, centers=occ_centers)
        
        cci.dict2cloud(p_val_file.format(subject, param_set_name), outdict)
        cci.dict2cloud(p_val_file.format(subject, param_set_name + "_occ" + str(occ_distance)), occ_outdict)
    else:
        print("NOT COMPLETE YET!!! Finish running the permutations first...")
        outdict = dict()

    return outdict

def spatial_perm_v_or_l_finished(subject, param_set_name):
    # Look at all individual line permutations to see if those are done
    filename_indiv = "spatial_perm_v_or_l_pval/{}_{}_{}_{}.ccd"
    indiv_files = [cci.glob(filename_indiv.format(subject, param_set_name, h, '*'))
                   for h in range(2)]
    filename_centers = "line_data/{}_{}_h{}.hf5/center_verts"
    centers = [cci.download_raw_array(filename_centers.format(subject, param_set_name, h))
               for h in range(2)]
    print("Number of lines to do:    {} {}".format(len(centers[0]), len(centers[1])))
    print("Number of lines finished: {} {}".format(len(indiv_files[0]), len(indiv_files[1])))
    complete = (len(centers[0])==len(indiv_files[0]) and len(centers[1])==len(indiv_files[1]))
    return complete
