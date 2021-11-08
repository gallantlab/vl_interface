import sys
import cortex
import numpy as np

from vl_interface import SUBJECTS, XFMS, CC_INTERFACE
from vl_interface.utils import load_model_info, mcorr

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)


if len(sys.argv)>1 and (sys.argv[1] in SUBJECTS):
    subjects = [sys.argv[1]]
    print("Got subject from command line: {}".format(subject[0]))
else:
    print("No subject provided... creating all plots")
    subjects = SUBJECTS

outfile = "../../figures/wt_corr/{0}_wt_corr_thresh.png"
plot_kwargs = dict(vmin=0, vmax=0.5, cmap="RdPu")
thresh = 0.0625

for subject in subjects:
    xfm = XFMS[subject]

    swt, mwt = load_model_info(subject)

    modelname = 'f1'
    perf_file = "{}data/{}_wholehead_eng1000_banded_{}_perf.hf5/{}"
    story_perf = cci.download_raw_array(perf_file.format("story", subject, modelname, 'corr'))[1]
    movie_perf = cci.download_raw_array(perf_file.format("movie", subject, modelname, 'corr'))[1]
    
    # Find projection of story and movie weights on templates
    wt_corr = mcorr(mwt, swt)
    wt_corr[wt_corr<thresh] = np.nan

    corr_vol = cortex.Volume(wt_corr, subject, xfm, **plot_kwargs)
    _ = cortex.quickflat.make_png(outfile.format(subject), corr_vol,
                                  with_colorbar=False, with_labels=False,
                                  with_curvature=True, linewidth=3)
