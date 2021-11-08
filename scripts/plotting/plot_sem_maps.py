import numpy as np
import cortex
import sys

from vl_interface import SUBJECTS, XFMS
from vl_interface.utils import load_model_info, load_pcs, scale_rgb_for_plotting

subject = sys.argv[1]
assert subject in SUBJECTS

if len(sys.argv)>2:
    model_type = sys.argv[2]
else:
    model_type = 'tikreg'

if len(sys.argv)>3:
    story_model = sys.argv[3]
else:
    story_model = 'story_f1'

if len(sys.argv)>4:
    movie_model = sys.argv[4]
else:
    movie_model = 'movie_f1'


outfile = "../../figures/{}_sem_map/{}_{}_{}.png"

pcs_used = [0,1,2]
flips = [1,2]

pcs = load_pcs()
pcs = pcs[pcs_used]
for f in flips:
    pcs[f] *= -1

xfm = XFMS[subject]
kwargs = dict(with_curvature=True, with_colorbar=False, with_labels=False)

swt, mwt = load_model_info(subject, model_type=model_type,
                           story_model=story_model, movie_model=movie_model)

story_rgb = np.dot(pcs, swt)
movie_rgb = np.dot(pcs, mwt)

story_map = scale_rgb_for_plotting(story_rgb)
movie_map = scale_rgb_for_plotting(movie_rgb)

story_red = cortex.Volume(story_map[0], subject, xfm)
story_green = cortex.Volume(story_map[1], subject, xfm)
story_blue = cortex.Volume(story_map[2], subject, xfm)
_ = cortex.quickflat.make_png(outfile.format("story", subject, model_type, story_model),
                              cortex.VolumeRGB(story_red, story_green, story_blue, subject, xfm),
                              **kwargs)

movie_red = cortex.Volume(movie_map[0], subject, xfm)
movie_green = cortex.Volume(movie_map[1], subject, xfm)
movie_blue = cortex.Volume(movie_map[2], subject, xfm)
_ = cortex.quickflat.make_png(outfile.format("movie", subject, model_type, movie_model),
                              cortex.VolumeRGB(movie_red, movie_green, movie_blue, subject, xfm),
                              **kwargs)
