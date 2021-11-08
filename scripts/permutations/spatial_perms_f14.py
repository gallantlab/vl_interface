import numpy as np
import sys
import cortex
import cortex.polyutils

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VL.permutations.spatial_perms_f14")

from vl_interface import SUBJECTS, XFMS, CC_INTERFACE
from vl_interface.line_io import recon_line
from vl_interface.line_permutation import spatial_perm_v_or_l
from vl_interface.utils import load_model_info

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)

subject = sys.argv[1]
assert subject in SUBJECTS

hem = int(sys.argv[2])
line_nums = np.array(sys.argv[3:], dtype=int)
xfm = XFMS[subject]

params = dict(name='f14', dist_from_line=5, ap_params=[10.,5.], m=1)

logger.info("Subject {}, param set {}".format(subject, params['name']))

# Loading semantic weights only once to speed things up
storydata_wt, moviedata_wt = load_model_info(subject, xfm, model_type='tikreg')

# Load center verts of all points and the verts that make up their best lines
center_verts = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/center_verts".format(subject, params['name'], h)) for h in range(2)]
line_verts = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/line_verts".format(subject, params['name'], h)) for h in range(2)]
grad_metrics = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/grad_metric".format(subject, params['name'], h)) for h in range(2)]
v_or_ls = [cci.download_raw_array("line_data/{}_{}_h{}.hf5/v_or_l".format(subject, params['name'], h)) for h in range(2)]

logger.info("Waiting to reconstruct all lines...")
# Reconstruct all lines
all_lines = [[recon_line(subject, xfm, params, h, center_verts[h][n], line_verts[h][n])
             for n in range(len(center_verts[h]))] for h in range(2)]
logger.info("Finished reconstructing all lines!!!")

# save pvals and line numbers together in chunks
outfile = "spatial_perm_v_or_l_pval/{}_{}_{}_chunk{}.hf5"

save_every = 500
lines_to_save = []
pvals_to_save = []

for line_num in line_nums:
    p_val = spatial_perm_v_or_l(subject, all_lines, hem, line_num, v_or_ls[hem][line_num],
                                        grad_metrics[hem][line_num], storydata_wt, moviedata_wt) 
    
    lines_to_save.append(line_num)
    pvals_to_save.append(p_val)
    if len(lines_to_save)==save_every:
        outdict = dict(center_verts=np.array(lines_to_save),
                       pvals=np.array(pvals_to_save))
        _ = cci.dict2cloud(outfile.format(subject, params['name'], hem,
                                          np.random.randint(10000,99999)),
                           outdict)
        logger.info("Uploaded information for {} lines".format(len(lines_to_save)))
        lines_to_save = []
        pvals_to_save = []

#save any leftovers
if len(lines_to_save)>0:
    outdict = dict(center_verts=np.array(lines_to_save),
                   pvals=np.array(pvals_to_save))
    _ = cci.dict2cloud(outfile.format(subject, params['name'], hem,
                                      np.random.randint(10000,99999)),
                       outdict)
    logger.info("Uploaded information for {} lines".format(len(lines_to_save)))
