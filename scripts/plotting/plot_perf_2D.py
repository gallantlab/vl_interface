import sys
import cortex
import numpy as np

from vl_interface import SUBJECTS, XFMS, CC_INTERFACE

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)


if len(sys.argv)>1 and (sys.argv[1] in SUBJECTS):
    subjects = [sys.argv[1]]
    print("Got subject from command line: {}".format(subject))
else:
    print("No subject provided... creating all plots")
    subjects = SUBJECTS


outfile = "../../figures/model_perf/{}_model_perf_{}.png"

plot_kwargs = dict(vmin=0, vmax=0.5, vmin2=0, vmax2=0.5, cmap="PU_RdBu_covar")

for subject in subjects:
    xfm = XFMS[subject]

    modelname = 'f1'
    perf_file = "{}data/{}_wholehead_eng1000_banded_{}_perf.hf5"
    story_perf = cci.cloud2dict(perf_file.format("story", subject, modelname))['corr'][1]
    movie_perf = cci.cloud2dict(perf_file.format("movie", subject, modelname))['corr'][1]
    
    perf_vol = cortex.Volume2D(story_perf, movie_perf, subject, xfm, **plot_kwargs)
    _ = cortex.quickflat.make_png(outfile.format(subject, modelname), perf_vol,
                                  with_colorbar=False, with_labels=False,
                                  with_curvature=True, linewidth=3)
