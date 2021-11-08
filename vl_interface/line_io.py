import numpy as np
import os
import subprocess as sp
import cortex
import cortex.polyutils

from vl_interface import CC_INTERFACE

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)

from vl_interface.line_objects import BrainLine
from vl_interface.utils import load_model_info


def recon_line(subject, xfm, param_set, hem, center_vert, line_verts):
    # Filter stupid warnings during line reconstruction about "(almost) singular matrix"
    import warnings
    warnings.filterwarnings("ignore")
    
    distance_bound = param_set['ap_params'][0]/2 + param_set['dist_from_line']
    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, 'fiducial')]

    try:
        patch = surfs[hem].get_geodesic_patch(center_vert, radius=distance_bound)
        subsurf = surfs[hem].create_subsurface(vertex_mask=patch['vertex_mask'])

        # Reconstruct line and ap_locs with info provided and subsurface
        mapped_verts = subsurf.subsurface_vertex_map[line_verts[np.isfinite(line_verts)].astype(int)]
        line = BrainLine(subject, xfm, hem, param_set, verts=mapped_verts)
        _ = line.get_ap_locs(subsurf, param_set['dist_from_line'])
        assert(line.native_surf) # Make sure we're back in native vertex space now
        _ = line.subdivide_center_vert(param_set['dist_from_line'],
                                       param_set['ap_params'],
                                       center_vert)
        if np.mean(line.subdiv_ap)==np.max(line.subdiv_ap):
            print("geodesic distances are not working well with subsurface")
            1/0
    except:
        print("Bad subsurface... manually creating from all geodesic distances")
        all_dists = surfs[hem].geodesic_distance([center_vert], m=2)
        vertex_mask = all_dists < distance_bound
        subsurf = surfs[hem].create_subsurface(vertex_mask=vertex_mask)

        # Reconstruct line and ap_locs with info provided and subsurface
        mapped_verts = subsurf.subsurface_vertex_map[line_verts[np.isfinite(line_verts)].astype(int)]
        # Something weird is happening
        error_val =  np.iinfo(np.int32).max
        mapped_verts = mapped_verts[mapped_verts < error_val]
        # Reconstruct line from vertices actually within the right radius
        line = BrainLine(subject, xfm, hem, param_set, verts=mapped_verts)
        _ = line.get_ap_locs(subsurf, param_set['dist_from_line'])
        assert(line.native_surf) # Make sure we're back in native vertex space now
        _ = line.subdivide_center_vert(param_set['dist_from_line'],
                                       param_set['ap_params'],
                                       center_vert)
    return line

def load_line_metadata(subject, param_set_name, xy=False):
    if xy:
        file_template = "line_data_xy/{subject}_{param_set_name}_h{hem}.hf5"
    else:
        file_template = "line_data/{subject}_{param_set_name}_h{hem}.hf5"
    output = [cci.cloud2dict(file_template.format(subject=subject,
                                                  param_set_name=param_set_name,
                                                  hem=h)) for h in range(2)]
    return output

