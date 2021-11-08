import sys
import os
import numpy as np
import cortex
from cortex.export import save_3d_views

from vl_interface import SUBJECTS, XFMS, CC_INTERFACE
from vl_interface import get_subjects, get_xfms
from vl_interface.utils import load_model_info, load_pcs, scale_rgb_for_plotting, fdr_correct
from vl_interface.line_io import recon_line

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)


if len(sys.argv)>1 and sys.argv[1] in SUBEJCTS:
    subject_list = [sys.argv[1]]
    print("Got subject from command line")
else:
    subject_list = SUBJECTS
    print("Doing all subjects")

xfm_list = [XFMS[s] for s in subject_list]

if len(sys.argv)>2:
    param_set_name = sys.argv[2]
    print("Got param set name from command line")
else:
    param_set_name = 'f13'
    print("Using default param set name: {}".format(param_set_name))

all_f_params = dict(f13=dict(name='f13', dist_from_line=10, ap_params=[25.,5.], m=1))

if param_set_name.startswith('f'):
    param_set = all_f_params[param_set_name]
else:
    Exception("param_set_name {} not supported".format(param_set_name))

do_inflated = False

for subject, xfm in zip(subject_list, xfm_list):
    if param_set_name=='f13':
        pval_dict = cci.cloud2dict("spatial_perm_v_or_l_pval/{}_{}.hf5".format(subject, param_set_name + "_occ50"))
    else:
        pval_dict = cci.cloud2dict("sem_perm_div_pval/{}_{}.hf5".format(subject, param_set_name + "_occ50"))
    p_val = pval_dict['p_val']
    p_centers = pval_dict['centers']

    which_fdr = 0 
    fdr_thresh = fdr_correct(p_val, 0.05)[which_fdr]
    print(fdr_thresh)

    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
    numl = surfs[0].pts.shape[0]
    numr = surfs[1].pts.shape[0]

    p_verts = [p_centers[p_centers<numl], p_centers[p_centers>=numl]-numl]

    valid_p_centers = p_val <= fdr_thresh
    border_verts = np.array(p_centers[valid_p_centers])
    border_verts = [border_verts[border_verts<numl], border_verts[border_verts>=numl]-numl]
    dists = [surfs[h].geodesic_distance(border_verts[h]) for h in range(2)]
    border_region = [np.where(dists[h]<5)[0] for h in range(2)]

    temp_border = np.zeros(numl+numr)
    for h in range(2):
        temp_border[border_region[h] + h*numl] = 1

    center_verts = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/center_verts".format(subject, param_set_name, h))
                    for h in range(2)]
    line_verts = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/line_verts".format(subject, param_set_name, h))
              for h in range(2)]
    grad_metric = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/grad_metric".format(subject, param_set_name, h))
                   for h in range(2)]

    # Make cached divisions first, using the save file
    outfile = "border_info_old/{}_{}.hf5"
    outdict = cci.cloud2dict(outfile.format(subject, param_set_name))
    vis_verts = outdict['vis_verts']
    ling_verts = outdict['ling_verts']
    mid_verts = outdict['mid_verts']

    # Get model data for plotting
    model_kwargs = dict(model_type='tikreg', story_model='story_f1', movie_model='movie_f1')
    storydata_wt, moviedata_wt = load_model_info(subject, xfm, **model_kwargs)
    
    pcs_used = [0,1,2]
    flips = [1,2]

    pcs = load_pcs()
    pcs = pcs[pcs_used]
    for f in flips:
        pcs[f] *= -1

    story_rgb = np.dot(pcs, np.hstack(storydata_wt))
    movie_rgb = np.dot(pcs, np.hstack(moviedata_wt))

    story_map = scale_rgb_for_plotting(story_rgb)
    movie_map = scale_rgb_for_plotting(movie_rgb)

    # Only get verts that can actually be plotted in the flatmap
    [lpt, lpoly], [rpt, rpoly] = cortex.db.get_surf(subject, "flat", nudge=True)
    allpt = np.vstack([lpt, rpt])
    plottable_verts = [np.unique(lpoly), np.unique(rpoly)]

    alpha = np.zeros(numl+numr)
    for h in range(2):
        for v in border_region[h]:
            if v in plottable_verts[h] and (vis_verts[v+h*numl] or ling_verts[v+h*numl]):
                alpha[v + h*numl] = 1
    
    combined_map = np.zeros_like(story_map)
    combined_map[:, ling_verts] = story_map[:, ling_verts]
    combined_map[:, vis_verts] = movie_map[:, vis_verts]
    combined_map[:, mid_verts] = 0

    # Have to do this again for weird pycortex reasons, or you get a white blob where it should be alpha-ed out
    for h in range(2):
        for v in border_region[h]:
            if v in plottable_verts[h] and (~vis_verts[v+h*numl] and ~ling_verts[v+h*numl]):
                combined_map[:, v + h*numl] = 0

    combined_red = cortex.Vertex(combined_map[0].astype(np.float32), subject)
    combined_green = cortex.Vertex(combined_map[1].astype(np.float32), subject)
    combined_blue = cortex.Vertex(combined_map[2].astype(np.float32), subject)
    outvol = cortex.VertexRGB(combined_red, combined_green, combined_blue, subject, alpha=alpha)

    fname = "../../figures/combined_sem_maps/{}_{}.png".format(subject, param_set_name)
    _ = cortex.quickflat.make_png(fname, outvol, with_colorbar=False, with_labels=False, with_curvature=True,
                                  cvmin=-5., cvmax=5., cvthr=True)

    if do_inflated:
        basepath = "../../figures/combined_sem_maps/"
        base_name = "{}_{}".format(subject, param_set_name)
        list_angles = ['lateral_pivot', 'medial_pivot', 'bottom_pivot']
        list_surfaces = ['inflated'] * 3
        _ = save_3d_views(outvol, os.path.join(basepath,base_name), list_angles=list_angles, list_surfaces=list_surfaces,
                          size=(1024*6, 784*4), viewer_params=dict(labels_visible=[], overlays_visible=[]))

    div_map = np.zeros_like(story_map)
    div_map[2, ling_verts] = 255.
    div_map[0, vis_verts] = 255.
    div_map[:, mid_verts] = 0.

    # Have to do this again for weird pycortex reasons, or you get a white blob where it should be alpha-ed out
    for h in range(2):
        for v in border_region[h]:
            if v in plottable_verts[h] and (~vis_verts[v+h*numl] and ~ling_verts[v+h*numl]):
                div_map[:, v + h*numl] = 0.

    div_red = cortex.Vertex(div_map[0].astype(np.float32), subject)
    div_green = cortex.Vertex(div_map[1].astype(np.float32), subject)
    div_blue = cortex.Vertex(div_map[2].astype(np.float32), subject)

    outvol = cortex.VertexRGB(div_red, div_green, div_blue, subject, alpha=alpha)
    fname = "../../figures/combined_sem_map_divs/{}_{}.png".format(subject, param_set_name)
    _ = cortex.quickflat.make_png(fname, outvol, with_colorbar=False, with_labels=False, with_curvature=True,
                                  cvmin=-5., cvmax=5., cvthr=True)

    if do_inflated:
        basepath = "../../figures/combined_sem_map_divs/"
        _ = save_3d_views(outvol, os.path.join(basepath,base_name), list_angles=list_angles, list_surfaces=list_surfaces,
                          size=(1024*6, 784*4), viewer_params=dict(labels_visible=[], overlays_visible=[]))
