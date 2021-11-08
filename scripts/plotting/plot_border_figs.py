import sys
import cortex
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm

from vl_interface import SUBJECTS, XFMS, CC_INTERFACE
from vl_interface.line_io import recon_line

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

distance_bound = param_set['ap_params'][0]/2 + param_set['dist_from_line']

with_dropout = True

for subject,xfm in zip(subject_list, xfm_list):
    cache_file = "border_info/{}_{}.hf5"
    print("Downloading cache file...")
    outdict = cci.cloud2dict(cache_file.format(subject, param_set_name))
    first_verts = outdict['first_verts'].astype(int)
    last_verts = outdict['last_verts'].astype(int)
    hems = outdict['hems'].astype(int)
    grad_metrics = outdict['grad_metrics']
    wt_corrs = outdict['wt_corrs']

    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
    numl = surfs[0].pts.shape[0]
    numr = surfs[1].pts.shape[0]
    numpts = numl + numr

    [lpt, lpoly], [rpt, rpoly] = cortex.db.get_surf(subject, "flat", nudge=True)
    allpt = np.vstack([lpt, rpt])
    plottable_verts = [np.unique(lpoly), np.unique(rpoly)]
    
    X = []
    Y = []
    U = []
    V = []
    val_grad = []
    val_corr = []

    for f,l,h,grad,corr in zip(first_verts, last_verts, hems, grad_metrics, wt_corrs):
        # Check which of these arrows are actually plottable on flatmaps
        if (f in plottable_verts[h]) and (l in plottable_verts[h]):
            X.append(allpt[f + h*numl,0])
            Y.append(allpt[f + h*numl,1])
            U.append(allpt[l + h*numl,0] - allpt[f + h*numl,0])
            V.append(allpt[l + h*numl,1] - allpt[f + h*numl,1])
            val_grad.append(grad)
            val_corr.append(corr)

    max_len = 70
    
    # First plot gradient metrics
    cmap = cm.get_cmap('Reds')
    cmap_flip = True
    if param_set_name in ['f5', 'f9']:
        abs_vmax = 0.0005
    elif param_set_name in ['f13', 'f15']:
        abs_vmax = 0.0001
    else:
        abs_vmax = 0.0002

    empty_vol = cortex.Vertex.empty(subject, value=np.nan)
    fig = cortex.quickshow(empty_vol, with_curvature=True, cvmin=-5., cvmax=5., cvthr=True,
                           with_colorbar=False, with_rois=True, with_labels=False, height=2048)

    for x,y,u,v,grad in zip(X,Y,U,V,val_grad):
        if np.sqrt(u**2+v**2) < max_len:
            w = np.min([1, grad/abs_vmax])
            if cmap_flip:
                s = cmap(1-w)
            else:
                s = cmap(w)
            plt.quiver(x,y,u,v, color=s, scale=1, units='xy', width=w, headwidth=10, edgecolor='k', lw=0.5*w+0.1,
                       zorder=np.round(grad/abs_vmax*100+10))

    if with_dropout:
        dropout_output = "dropout/{}_apply_avg.ccd".format(subject)
        dropout_vox = cci.download_raw_array(dropout_output)
        dropout_vol = cortex.Volume(dropout_vox<15, subject, xfm)
        hatch = cortex.quickflat.composite.add_hatch(fig, dropout_vol, hatch_color=(0.1, 0., 0.3))
        plt.savefig("../../figures/sem_perm_sig_long/{}_{}_dropout.png".format(subject, param_set_name))
    else:
        plt.savefig("../../figures/sem_perm_sig_long/{}_{}.png".format(subject, param_set_name))

    plt.close('all')


    # Then plot the wt corr 
    scale_factor = 0.6
    cmap_pos = cm.get_cmap('Reds')
    cmap_neg = cm.get_cmap('Blues')
    abs_vmax = 0.3
    
    empty_vol = cortex.Vertex.empty(subject, value=np.nan)
    fig = cortex.quickshow(empty_vol, with_curvature=True, cvmin=-5., cvmax=5., cvthr=True,
                           with_colorbar=False, with_rois=True, with_labels=False, height=2048)

    for x,y,u,v,corr in zip(X,Y,U,V,val_corr):
        if np.sqrt(u**2+v**2) < max_len and corr > -10:
            if corr >= 0:
                w = np.min([1, corr/abs_vmax])
                s = cmap_pos(scale_factor*w)
            else:
                w = np.max([-1, corr/abs_vmax])
                s = cmap_neg(-1*scale_factor*w)
            plt.quiver(x,y,u,v, color=s, scale=1, units='xy', width=np.abs(w), headwidth=10, edgecolor='k', lw=0.5*w+0.1,
                       zorder=np.round(np.abs(corr/abs_vmax*100)+10))

    if with_dropout:
        dropout_output = "dropout/{}_apply_avg.ccd".format(subject)
        dropout_vox = cci.download_raw_array(dropout_output)
        dropout_vol = cortex.Volume(dropout_vox<15, subject, xfm)
        hatch = cortex.quickflat.composite.add_hatch(fig, dropout_vol, hatch_color=(0.1, 0., 0.3))
        plt.savefig("../../figures/wt_corr_band_sig_long/{}_{}_dropout.png".format(subject, param_set_name))
    else:
        plt.savefig("../../figures/wt_corr_band_sig_long/{}_{}.png".format(subject, param_set_name))

    plt.close('all')
