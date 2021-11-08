import cortex
import sys

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VL.parameters.f15")

from vl_interface import SUBJECTS, XFMS, CC_INTERFACE
from vl_interface.draw_lines import all_angles
from vl_interface.utils import load_model_info

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)

subject = sys.argv[1]
assert subject in SUBJECTS

hem = int(sys.argv[2])
verts = np.array(sys.argv[3:], dtype=int)
xfm = XFMS[subject]

params = dict(name='f15', dist_from_line=5, ap_params=[25.,5.], m=1)

# Faster to load and map voxel weights only once
logger.info("Loading model info...")
model_type = 'tikreg'
storydata_wt, moviedata_wt = load_model_info(subject, xfm, model_type=model_type)

outfile = "temp_line_data/{}_{}_{}_{}.hf5/wt_corr"

logger.info("Doing analysis for {}, param set {}".format(subject, params['name']))
for vert in verts:
    # Check for existance of file in case job was interrupted and restarting
    if not cci.exists_object(outfile.format(subject, params['name'], hem, vert)):
        try:
            _,best_line = all_angles(subject, xfm, hem, vert, params, storydata_wt, moviedata_wt)
            logger.info("Job completed: {} {}".format(hem, vert))
        except:
            logger.info("There is a problem with this vertex: {} {}".format(hem, vert))
    else:
        logger.info("Job already completed: {} {}".format(hem, vert))
