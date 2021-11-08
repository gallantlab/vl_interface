import sys
import cortex
import numpy as np

from matplotlib import cm
cmap = cm.get_cmap('Reds')
cmap_flip = True

from vl_interface import SUBJECTS, XFMS, CC_INTERFACE
from vl_interface.line_io import recon_line
from vl_interface.utils import load_model_info, fdr_correct

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)


if len(sys.argv)>1 and sys.argv[1] in SUBJECTS:
    subject_list = [sys.argv[1]]
    print("Got subject from command line")
else:
    subject_list = SUBJECTS
    print("Doing all subjects")
    print(subject_list)

xfm_list = [XFMS[s] for s in subject_list]

if len(sys.argv)>2:
    param_set_name = sys.argv[2]
    print("Got param set name from command line")
else:
    param_set_name = 'f13'
    print("Using default param set name: {}".format(param_set_name))

all_f_params = dict(f13=dict(name='f13', dist_from_line=10, ap_params=[25.,5.], m=1),
                    f14=dict(name='f14', dist_from_line=5, ap_params=[10.,5.], m=1),
                    f15=dict(name='f15', dist_from_line=5, ap_params=[25.,5.], m=1))

if param_set_name.startswith('f'):
    param_set = all_f_params[param_set_name]
else:
    Exception("param_set_name {} not supported".format(param_set_name))


for subject,xfm in zip(subject_list, xfm_list):
    center_verts = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/center_verts".format(subject, param_set_name, h))
                    for h in range(2)]
    line_verts = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/line_verts".format(subject, param_set_name, h))
              for h in range(2)]
    grad_metric = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/grad_metric".format(subject, param_set_name, h))
                   for h in range(2)]
    wt_corr = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/wt_corr".format(subject, param_set_name, h))
               for h in range(2)]

    if param_set_name in ['f13', 'f14', 'f15']:
        v_or_l =  [cci.download_raw_array("line_data/{}_{}_h{}.hf5/v_or_l".format(subject, param_set_name, h))
                   for h in range(2)]
        pval_dict = cci.cloud2dict("spatial_perm_v_or_l_pval/{}_{}.hf5".format(subject, param_set_name + "_occ50"))
    else:
        pval_dict = cci.cloud2dict("sem_perm_div_pval/{}_{}.hf5".format(subject, param_set_name + "_occ50"))
    p_val = pval_dict['p_val']
    p_centers = pval_dict['centers']

    which_fdr = 0
    fdr_thresh = fdr_correct(p_val, 0.05)[which_fdr]
    
    print("p-value threshold after fdr_correction: {}".format(fdr_thresh))
    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
    numl = surfs[0].pts.shape[0]
    numr = surfs[1].pts.shape[0]

    valid_p_centers = p_val <= fdr_thresh
    border_verts = np.array(p_centers[valid_p_centers])
    border_verts = [border_verts[border_verts<numl], border_verts[border_verts>=numl]-numl]
    dists = [surfs[h].geodesic_distance(border_verts[h]) for h in range(2)]
    border_region = [np.where(dists[h]<5)[0] for h in range(2)]

    temp_border = np.zeros(numl+numr)
    for h in range(2):
        temp_border[border_region[h] + h*numl] = 1

    # Stuff for arrow plots
    n_sig = np.sum(p_val <= fdr_thresh)
    first_verts = np.zeros(n_sig)
    last_verts = np.zeros(n_sig)
    hems = np.zeros(n_sig)
    grad_metrics = np.zeros(n_sig)
    wt_corrs = np.zeros(n_sig)
    
    if param_set_name in ['f13', 'f14', 'f15']:
        v_or_ls = []

    all_lines = []
    sig_count = 0

    # Also adding stuff for combined sem maps
    posterior_verts = np.zeros(numl+numr)
    anterior_verts = np.zeros(numl+numr)
    posterior_verts_scaled = np.zeros(numl+numr)
    anterior_verts_scaled = np.zeros(numl+numr)

    for n,center in enumerate(p_centers):
        if p_val[n] <= fdr_thresh:
            if center >= numl:
                h = 1
                h_center = center - numl
            else:
                h = 0
                h_center = center
            idx = np.where(center_verts[h]==h_center)[0][0]
            curr_line_verts = line_verts[h][idx]
            curr_line_verts = curr_line_verts[np.isfinite(curr_line_verts)].astype(int)

            line = recon_line(subject, xfm, param_set, h, h_center, curr_line_verts)
            all_lines.append(line)
            verts_in_subdiv = [v for v in curr_line_verts if v in line.subdiv_verts]
            first_verts[sig_count] = verts_in_subdiv[0]
            last_verts[sig_count] = verts_in_subdiv[-1]
            hems[sig_count] = h
            grad_metrics[sig_count] = grad_metric[h][idx]
            wt_corrs[sig_count] = wt_corr[h][idx]

            if param_set_name in ['f13', 'f14', 'f15']:
                v_or_ls.append(v_or_l[h][idx])
            
            sig_count += 1
            
            # Split in the middle of the subdivision like before
            min_ap = min(line.subdiv_ap)
            ant_v = np.array(line.subdiv_verts)[line.subdiv_ap > param_set['ap_params'][0]/2 + min_ap]
            pos_v = np.array(line.subdiv_verts)[line.subdiv_ap < param_set['ap_params'][0]/2 + min_ap]
            anterior_verts[ant_v + h*numl] += 1
            posterior_verts[pos_v + h*numl] += 1
            anterior_verts_scaled[ant_v + h*numl] += grad_metric[h][idx]
            posterior_verts_scaled[pos_v + h*numl] += grad_metric[h][idx]
            
            if (sig_count)%500==0:
                print("{} out of {} sig arrows done".format(sig_count, n_sig))

    # Divide vertices into visual and lingusitic, though can be in both
    vis_verts = posterior_verts >= anterior_verts
    vis_verts[~temp_border.astype(bool)] = False
    vis_verts[np.logical_and(posterior_verts==0, anterior_verts==0)] = False
    ling_verts = anterior_verts >= posterior_verts
    ling_verts[~temp_border.astype(bool)] = False
    ling_verts[np.logical_and(posterior_verts==0, anterior_verts==0)] = False

    mid_verts = np.zeros(numl+numr).astype(bool)
    vis_idx = np.where(vis_verts)[0]
    ling_idx = np.where(ling_verts)[0]
    # First check for vision verts with neighbors that are language verts
    for v in vis_idx:
        if v<numl:
            h = 0
            h_center = v
        else:
            h = 1
            h_center = v-numl
        neighbors = [n for n in surfs[h].graph.neighbors(h_center)]
        is_border = np.array([n+h*numl in ling_idx for n in neighbors]).any()
        if is_border:
            mid_verts[h_center+h*numl] = True

    # Then check for language verts with neighbors that are vision verts
    for v in ling_idx:
        if v<numl:
            h = 0
            h_center = v
        else:
            h = 1
            h_center = v-numl
        neighbors = [n for n in surfs[h].graph.neighbors(h_center)]
        is_border = np.array([n+h*numl in vis_idx for n in neighbors]).any()
        if is_border:
            mid_verts[h_center+h*numl] = True
    
    # Divide vertices into visual and lingusitic, though can be in both
    vis_verts_scaled = posterior_verts_scaled >= anterior_verts_scaled
    vis_verts_scaled[~temp_border.astype(bool)] = False
    vis_verts_scaled[np.logical_and(posterior_verts_scaled==0, anterior_verts_scaled==0)] = False
    ling_verts_scaled = anterior_verts_scaled >= posterior_verts_scaled
    ling_verts_scaled[~temp_border.astype(bool)] = False
    ling_verts_scaled[np.logical_and(posterior_verts_scaled==0, anterior_verts_scaled==0)] = False

    mid_verts_scaled = np.zeros(numl+numr).astype(bool)
    vis_idx_scaled = np.where(vis_verts_scaled)[0]
    ling_idx_scaled = np.where(ling_verts_scaled)[0]
    # First check for vision verts with neighbors that are language verts
    for v in vis_idx_scaled:
        if v<numl:
            h = 0
            h_center = v
        else:
            h = 1
            h_center = v-numl
        neighbors = [n for n in surfs[h].graph.neighbors(h_center)]
        is_border = np.array([n+h*numl in ling_idx for n in neighbors]).any()
        if is_border:
            mid_verts_scaled[h_center+h*numl] = True

    # Then check for language verts with neighbors that are vision verts
    for v in ling_idx_scaled:
        if v<numl:
            h = 0
            h_center = v
        else:
            h = 1
            h_center = v-numl
        neighbors = [n for n in surfs[h].graph.neighbors(h_center)]
        is_border = np.array([n+h*numl in vis_idx for n in neighbors]).any()
        if is_border:
            mid_verts_scaled[h_center+h*numl] = True
    
    outdict = dict(first_verts=first_verts,
                   last_verts=last_verts,
                   hems=hems,
                   grad_metrics=grad_metrics,
                   wt_corrs=wt_corrs,
                   vis_verts=vis_verts,
                   ling_verts=ling_verts,
                   mid_verts=mid_verts,
                   vis_verts_scaled=vis_verts_scaled,
                   ling_verts_scaled=ling_verts_scaled,
                   mid_verts_scaled=mid_verts_scaled)

    cache_file = "border_info_old/{}_{}.hf5"
    _ = cci.dict2cloud(cache_file.format(subject, param_set_name), outdict)
