import cortex
import cortex.polyutils
import numpy as np
from itertools import groupby
from scipy.stats import linregress

from vl_interface import CC_INTERFACE, XFMS

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)

from vl_interface.utils import (geodesic_path,
                                 anti_geodesic_path,
                                 tissots_subset,
                                 get_fovea_verts,
                                 check_and_flip,
                                 load_model_info)
from vl_interface.line_io import (save_lines,
                                   save_brain,
                                   recon_line,
                                   load_line_metadata)
from vl_interface.line_objects import (BrainLine, SpaghettiBrain)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VL.draw_lines")


def save_tissot(subject, xfm, param_set_name, distance=2*4.13):
    surfs = [cortex.polyutils.Surface(*d)
             for d in cortex.db.get_surf(subject, 'fiducial')]
    numl = surfs[0].pts.shape[0]
    
    print("Finding vertices within {}mm of sig voxels".format(distance))
    sig_verts = get_sig_verts(subject, xfm, distance)
    sig_verts = [sig_verts[:numl], sig_verts[numl:]]
    sig_verts = [np.where(sig==1)[0] for sig in sig_verts]
    print("Evenly sampling those vertices")
    tissots, centers = tissots_subset(subject, sig_verts, spacing=2.5)

    filename_tiss = "tissot/{0}_{1}_{2}_h{3}.ccd"
    for h in range(2):
        cci.upload_raw_array(filename_tiss.format(subject, param_set_name, distance, h),
                             np.array(centers[h]))


def all_angles(subject, xfm, h, vert, param_set, storydata_wt, moviedata_wt,
               save_dict=True, wt_corr=True):
    """This version should be used if the entire cortical surface is available.

    If using the data accompanying the publication, instead use all_angles_xy,
    as this is compatible with the xy coordinates of each vertex in flatmap space.

    This version is slightly more accurate, but both give similar results in 
    cortical locations that are not near the edges of the flatmap.
    """
    surfs = [cortex.polyutils.Surface(*d)
             for d in cortex.db.get_surf(subject, "fiducial")]
    numl = surfs[0].pts.shape[0]

    # Create subsurface around vert
    distance_bound = param_set['ap_params'][0]/2 + param_set['dist_from_line']
    try:
        patch = surfs[h].get_geodesic_patch(vert, radius=distance_bound)
        subsurf = surfs[h].create_subsurface(vertex_mask=patch['vertex_mask'])
        border_pts = np.where(subsurf.boundary_vertices)[0]

        all_lines = []
        for pt in border_pts:
            all_lines.append(BrainLine(subject, xfm, h, param_set))
            all_lines[-1].draw_line(pt, subsurf.subsurface_vertex_map[vert],
                                    border_pts, subsurf, param_set['m'])
            # If line drawing failed, remove
            if all_lines[-1].verts is None:
                _ = all_lines.pop()
        print("Lines drawn through vertex {}".format(vert))

        _ = [all_lines[l].get_ap_locs(subsurf, param_set['dist_from_line'])
             for l in range(len(all_lines))]
        print("AP locations of lines found")
        
        # Make special single subdivision
        _ = [all_lines[l].subdivide_center_vert(param_set['dist_from_line'],
                                                param_set['ap_params'],
                                                vert)
             for l in range(len(all_lines))]
        print("Centered ROIs around center vert")
        
        _ = [all_lines[l].semantic_analysis(param_set['dist_from_line'],
                                                    param_set['ap_params'],
                                                    storydata_wt, moviedata_wt)
             for l in range(len(all_lines))]
        print("Semantic analysis complete")
     
        # Now find just best line to save
        slope_diffs = np.array([line.slope_diffs for line in all_lines])
        line_id = np.argmax(slope_diffs)
        best_line = all_lines[line_id]
        print("Found best line")
        
    # Manually get subsurf mask if something is strange at that region of cortex
    except:
        print("Bad subsurface... manually creating from all geodesic distances")
        all_dists = surfs[h].geodesic_distance([vert], m=2)
        vertex_mask = all_dists < distance_bound
        subsurf = surfs[h].create_subsurface(vertex_mask=vertex_mask)
        border_pts = np.where(subsurf.boundary_vertices)[0]

        all_lines = []
        for pt in border_pts:
            all_lines.append(BrainLine(subject, xfm, h, param_set))
            all_lines[-1].draw_line(pt, subsurf.subsurface_vertex_map[vert],
                                    border_pts, subsurf, param_set['m'])
            # If line drawing failed, remove
            if all_lines[-1].verts is None:
                _ = all_lines.pop()
        print("Lines drawn through vertex {}".format(vert))

        _ = [all_lines[l].get_ap_locs(subsurf, param_set['dist_from_line'])
             for l in range(len(all_lines))]
        print("AP locations of lines found")
        
        # Make special single subdivision
        _ = [all_lines[l].subdivide_center_vert(param_set['dist_from_line'],
                                                param_set['ap_params'],
                                                vert)
             for l in range(len(all_lines))]
        print("Centered ROIs around center vert")

        _ = [all_lines[l].semantic_analysis(param_set['dist_from_line'],
                                                    param_set['ap_params'],
                                                    storydata_wt, moviedata_wt)
             for l in range(len(all_lines))]
        print("Semantic analysis complete")
     
        # Now find just best line to save
        slope_diffs = np.array([line.slope_diffs for line in all_lines])
        line_id = np.argmax(slope_diffs)
        best_line = all_lines[line_id]
        print("Found best line")

    return all_lines, best_line

def all_angles_xy(subject, xfm, h, vert, param_set, storydata_wt, moviedata_wt):
    """This version should be used if the entire cortical surface is not available,
    otherwise all_angles should be used.

    This version is slightly less accurate, but both give similar results in 
    cortical locations that are not near the edges of the flatmap.
    """
    cci_vl = cc.get_interface("spopham_vl_data", verbose=False)
    hems = ['lh', 'rh']
    fname = "mappers/{}_vertex_coords_{}.ccd".format(subject[:2], hems[h])   
    all_coords = cci_vl.download_raw_array(fname)

    vert_coords = all_coords[vert]
    if np.isnan(vert_coords).all():
        raise ValueError("Vertex coordinates are np.nan, they are not on the flatmap")

    distance_bound = param_set['ap_params'][0]/2 + param_set['dist_from_line']
    dists = np.linalg.norm(all_coords - vert_coords, axis=1)
    close_verts = np.where(dists < distance_bound)[0]
    close_coords = all_coords[close_verts]

    angles = np.linspace(0, np.pi, 181)[:-1]
    line_pts = np.arange(0, 2*distance_bound+0.01, 0.1) - distance_bound
    line_xs = np.array([np.cos(a) * line_pts + vert_coords[0] for a in angles])
    line_ys = np.array([np.sin(a) * line_pts + vert_coords[1] for a in angles])
   
    # Distance from any point that line should not continue to be drawn
    dist_thresh = 2

    all_line_verts = []
    # Get points closest to each line
    for xs,ys in zip(line_xs, line_ys):
        dists_to_line = np.array([np.linalg.norm(close_coords-np.array([x,y]), axis=1)
                                  for x,y in zip(xs,ys)])
        closest_idx = np.argmin(dists_to_line, axis=1)
        closest_dists = np.min(dists_to_line, axis=1)
        closest_idx = closest_idx[closest_dists <= dist_thresh]
        idx = np.array([k for k,g in groupby(closest_idx)])
        all_line_verts.append(idx)

    all_lines = [BrainLine(subject, xfm, h, param_set, verts=verts)
                 for verts in all_line_verts]

    smooth_factor = 2.0
    normsum = lambda v: v / v.sum(0)
    
    for line in all_lines:
        all_dists = np.array([np.linalg.norm(close_coords - close_coords[v], axis=1) for v in line.verts])
        min_dists = np.min(all_dists, axis=0)
        line.ROI_verts = np.where(min_dists < param_set['dist_from_line'])[0]
        ROI_dists_subset = np.array([[all_dists[v][p] for p in line.ROI_verts] for v in range(len(line.verts))])
        line_interdists = np.zeros(len(line.verts)-1)
        for num,v in enumerate(line.verts[:-1]):
            line_interdists[num] = np.linalg.norm(close_coords[v] - close_coords[line.verts[num+1]])
        ROI_vert_ap_locs = np.hstack([[0], np.cumsum(line_interdists)])
        ROI_wts = normsum(np.exp(-ROI_dists_subset / smooth_factor))
        line.ap_locs = np.dot(ROI_vert_ap_locs, ROI_wts)

    # Go back to full surface indices instead of "close" indices
    # Similar to BrainLine.subsurf_to_native()
    for line in all_lines:
        line.verts = close_verts[line.verts]
        line.ROI_verts = close_verts[line.ROI_verts]

    for line in all_lines:
        _ = line.subdivide_center_vert(param_set['dist_from_line'], param_set['ap_params'], vert)
        _ = line.semantic_analysis(param_set['dist_from_line'], param_set['ap_params'],
                                   storydata_wt, moviedata_wt)

    # Now find just best line to save
    slope_diffs = np.array([line.slope_diffs for line in all_lines])
    line_id = np.argmax(slope_diffs)
    best_line = all_lines[line_id]
    print("Found best line")
    
    return all_lines, best_line


def gather_all_angles(subject, param_set_name, centers_param_set_name='f4', check_dict=True):
    if check_dict:
        filename_line = "temp_line_data/{}_{}_{}_{}.hf5"
        filename_verts = "temp_line_data/{}_{}_{}_{}.hf5/line_verts"
    else:
        filename_line = "temp_lines/{}_{}_{}_{}.pkl"

    all_lines = [[] for h in range(2)]
    outfile = "line_data/{}_{}_h{}.hf5"
    print("Downloading from {}...".format(filename_line))
    if check_dict:
        for h in range(2):
            all_files = cci.glob(filename_verts.format(subject, param_set_name, h, '*'))
            line_nums = np.array([int(f.split('_')[-2][:-9]) for f in all_files])
            line_nums = np.sort(line_nums)
            n_lines = len(line_nums)
            print(subject, param_set_name, h, n_lines)

            for num,vert in enumerate(line_nums):
                all_lines[h].append(cci.cloud2dict(filename_line.format(subject, param_set_name, h, vert),
                                                   verbose=False))
                if num%1000==0:
                    print("{} out of {} downloaded for hem {}...".format(num, len(line_nums), h))
            print("Downloaded all for hem{}!!!".format(h))

            grad_metric = np.array([all_lines[h][idx]['grad_metric'].max() for idx in range(n_lines)])
            wt_corr = np.array([all_lines[h][idx]['wt_corr'].max() for idx in range(n_lines)])
            v_or_l = np.array([all_lines[h][idx]['v_or_l'].data[0] for idx in range(n_lines)])
            print("Aggregated all metrics, wt_corr, and v_or_l values...")
            
            line_lengths = np.array([len(all_lines[h][idx]['line_verts']) for idx in range(n_lines)])
            max_length = line_lengths.max()
            print("Found max line length...")
            
            line_verts = np.full((n_lines, max_length), np.nan)
            print(line_verts.shape)
            print("Going to aggregate all line verts...")
            for idx in range(n_lines):
                line_verts[idx, :line_lengths[idx]] = all_lines[h][idx]['line_verts']
                if idx%1000==0:
                    print("{} out of {} for hem {}...".format(idx, len(line_nums), h))
            print("Okay... about to upload... we'll see...")

            outdict = dict(center_verts=line_nums,
                           grad_metric=grad_metric,
                           wt_corr=wt_corr,
                           v_or_l=v_or_l,
                           line_verts=line_verts)
            cci.dict2cloud(outfile.format(subject, param_set_name, h), outdict)
            print("Uploaded all info for hem {}!!!".format(h))

    else:
        for h in range(2):
            all_files = cci.glob(filename_line.format(subject, param_set_name, h, '*'))
            line_nums = np.array([int(f.split('_')[-1][:-4]) for f in all_files])
            line_nums = np.sort(line_nums)
            n_lines = len(line_nums)
            print(subject, param_set_name, h, n_lines)
            for vert in line_nums:
                all_lines[h].append(cci.download_pickle(filename_line.format(subject, param_set_name, h, vert)))

            grad_metric = np.array([all_lines[h][idx].slope_diffs for idx in range(n_lines)])
            line_lengths = np.array([len(all_lines[h][idx].verts) for idx in range(n_lines)])
            max_length = line_lengths.max()

            line_verts = np.full((n_lines, max_length), np.nan)
            print(line_verts.shape)
            for idx in range(n_lines):
                line_verts[idx, :line_lengths[idx]] = all_lines[h][idx].verts

            outdict = dict(center_verts=line_nums,
                           grad_metric=grad_metric,
                           line_verts=line_verts)
            cci.dict2cloud(outfile.format(subject, param_set_name, h), outdict)
        
        # Also save all semantic vectors for ease of loading for permutations
        sem_vectors = np.array([l.sem_vectors for hem in all_lines for l in hem])
        sem_file = "sem_vectors/perm_{}_{}.ccd".format(subject, param_set_name)
        cci.upload_raw_array(sem_file, sem_vectors)
        print("Saved all semantic vectors for permutation tests...")

        save_lines(all_lines)
        print("Lines all gathered and saved...")

    return all_lines

def gather_all_angles_xy(subject, param_set, centers_param_set='f4'):
    param_set_name = param_set['name']
    filename_line = "temp_line_data_xy/{}_{}_{}_{}.hf5"
    filename_verts = "temp_line_data_xy/{}_{}_{}_{}.hf5/line_verts"

    all_lines = [[] for h in range(2)]
    outfile = "line_data_xy/{}_{}_h{}.hf5"
    print("Downloading from {}...".format(filename_line))
    for h in range(2):
        all_files = cci.glob(filename_verts.format(subject, param_set_name, h, '*'))
        line_nums = np.array([int(f.split('_')[-2][:-9]) for f in all_files])
        line_nums = np.sort(line_nums)
        n_lines = len(line_nums)
        print(subject, param_set_name, h, n_lines)
        for num,vert in enumerate(line_nums):
            all_lines[h].append(cci.cloud2dict(filename_line.format(subject, param_set_name, h, vert),
                                               verbose=False))
            if num%1000==0:
                print("{} out of {} downloaded for hem {}...".format(num, len(line_nums), h))

        grad_metric = np.array([all_lines[h][idx]['grad_metric'].max() for idx in range(n_lines)])
        line_lengths = np.array([len(all_lines[h][idx]['line_verts']) for idx in range(n_lines)])
        max_length = line_lengths.max()

        line_verts = np.full((n_lines, max_length), np.nan)
        print(line_verts.shape)
        for idx in range(n_lines):
            line_verts[idx, :line_lengths[idx]] = all_lines[h][idx]['line_verts']

        outdict = dict(center_verts=line_nums,
                       grad_metric=grad_metric,
                       line_verts=line_verts)
        cci.dict2cloud(outfile.format(subject, param_set_name, h), outdict)

    # Also save all semantic vectors for ease of loading for permutations
    sem_vectors = np.array([l['sem_vector'] for hem in all_lines for l in hem])
    sem_file = "sem_vectors_xy/perm_{}_{}.ccd".format(subject, param_set['name'])
    cci.upload_raw_array(sem_file, sem_vectors)
    print("Saved all semantic vectors for permutation tests...")

    # Only delete if all lines have been found, compare against tissot centers
    if centers_param_set is None:
        centers_param_set = param_set_name[:2]

    if angles_xy_finished(subject, param_set_name, centers_param_set):
        to_delete = cci.glob(filename_line.format(subject, param_set_name, '*', '*') + '*')
        for n,t in enumerate(to_delete):
            cci.rm(t)
            if n%500==0:
                print("{0} of {1} temp files deleted...".format(n, len(to_delete)))

    return all_lines


def angles_finished(subject, param_set_name, center_param_set_name='f4', distance=0, check_dict=True):
    if check_dict:
        filename_line = "temp_line_data/{}_{}_{}_{}.hf5/wt_corr"
        line_files = [cci.glob(filename_line.format(subject, param_set_name, h, '*')) for h in range(2)] 
    else:
        filename_line = "temp_lines/{}_{}_{}_{}.pkl"
        line_files = [cci.glob(filename_line.format(subject, param_set_name, h, '*')) for h in range(2)]
    tissot_file = "tissot/{}_{}_{}_h{}.ccd"

    if center_param_set_name is None:
        center_param_set_name = param_set_name

    centers = [cci.download_raw_array(tissot_file.format(subject, center_param_set_name[:2], distance, h))
               for h in range(2)]
    print("Number of lines to do:    {} {}".format(len(centers[0]), len(centers[1])))
    print("Number of lines finished: {} {}".format(len(line_files[0]), len(line_files[1])))
    return (len(centers[0])==len(line_files[0]) and len(centers[1])==len(line_files[1]))

def run_unfinished_angles(subject, params, center_param_set_name='f4', distance=0, check_dict=True):
    if angles_finished(subject, params['name'], center_param_set_name=center_param_set_name,
                       distance=distance, check_dict=check_dict):
        print("Already finished")
        return []
    else:
        param_set_name = params['name']
        if check_dict:
            filename_line = "temp_line_data/{}_{}_{}_{}.hf5/line_verts"
            line_files = [cci.glob(filename_line.format(subject, param_set_name, h, '*')) for h in range(2)]
            done_verts = [np.sort([int(f.split('_')[-2][:-9]) for f in files]) for files in line_files]
        else:
            filename_line = "temp_lines/{0}_{1}_{2}_{3}.pkl"
            line_files = [cci.glob(filename_line.format(subject, param_set_name, h, '*')) for h in range(2)]
            done_verts = [np.sort([int(f.split('_')[-1][:-4]) for f in files]) for files in line_files]

        tissot_file = "tissot/{}_{}_{}_h{}.ccd"
        if center_param_set_name is None:
            center_param_set_name = param_set_name
        centers = [cci.download_raw_array(tissot_file.format(subject, center_param_set_name[:2], distance, h))
                   for h in range(2)]

        incomplete_verts = [[] for h in range(2)]
        for h in range(2):
            for v in centers[h]:
                if v not in done_verts[h]:
                    incomplete_verts[h].append(v)
        print("Number of lines left to do: {}   {}".format(len(incomplete_verts[0]), len(incomplete_verts[1])))

        # Faster to load and map voxel weights only once
        print("Loading model info...")
        model_kwargs = dict(model_type='tikreg', story_model='story_f1', movie_model='movie_f1')
        
        xfm = XFMS[subject]
        storydata_wt, moviedata_wt = load_model_info(subject, xfm=xfm, **model_kwargs)

        for h in range(2):
            for vert in incomplete_verts[h]:
                print("Running job: {} {}".format(h, vert))
                _ = all_angles(subject, xfm, h, vert, params, storydata_wt, moviedata_wt)

def angles_xy_finished(subject, param_set_name, center_param_set_name='f4', distance=0):
    filename_line = "temp_line_data_xy/{}_{}_{}_{}.hf5/line_verts"
    line_files = [cci.glob(filename_line.format(subject, param_set_name, h, '*')) for h in range(2)]
    tissot_file = "tissot/{}_{}_{}_h{}.ccd"

    if center_param_set_name is None:
        center_param_set_name = param_set_name

    centers = [cci.download_raw_array(tissot_file.format(subject, center_param_set_name[:2], distance, h))
               for h in range(2)]

    cci_vl = cc.get_interface("spopham_vl_data", verbose=False)
    hems = ['lh', 'rh']
    fname = "mappers/{}_vertex_coords_{}.ccd"
    all_coords = [cci_vl.download_raw_array(fname.format(subject[:2], hems[h])) for h in range(2)]

    vert_coords = [all_coords[h][centers[h]] for h in range(2)]
    valid_centers = [np.isfinite(v).all(axis=1) for v in vert_coords]

    print("Number of lines:          {} {}".format(len(centers[0]), len(centers[1])))
    print("Number of VALID lines:    {} {}".format(valid_centers[0].sum(), valid_centers[1].sum()))
    print("Number of lines finished: {} {}".format(len(line_files[0]), len(line_files[1])))
    return (len(line_files[0])==valid_centers[0].sum() and len(line_files[1])==valid_centers[1].sum())

def run_unfinished_angles_xy(subject, params, center_param_set_name='f4', distance=0):
    if angles_xy_finished(subject, params['name'], center_param_set_name=center_param_set_name,
                          distance=distance):
        print("Already finished")
        return []
    else:
        param_set_name = params['name']
        filename_line = "temp_line_data/{}_{}_{}_{}.hf5/line_verts"
        line_files = [cci.glob(filename_line.format(subject, param_set_name, h, '*')) for h in range(2)]
        done_verts = [np.sort([int(f.split('_')[-2][:-9]) for f in files]) for files in line_files]

        tissot_file = "tissot/{}_{}_{}_h{}.ccd"
        if center_param_set_name is None:
            center_param_set_name = param_set_name
        centers = [cci.download_raw_array(tissot_file.format(subject, center_param_set_name[:2], distance, h))
                   for h in range(2)]

        cci_vl = cc.get_interface("spopham_vl_data", verbose=False)
        hems = ['lh', 'rh']
        fname = "mappers/{}_vertex_coords_{}.ccd"
        all_coords = [cci_vl.download_raw_array(fname.format(subject[:2], hems[h])) for h in range(2)]

        vert_coords = [all_coords[h][centers[h]] for h in range(2)]
        valid_centers = [np.isfinite(v).all(axis=1) for v in vert_coords]

        incomplete_verts = [[] for h in range(2)]
        for h in range(2):
            for v in centers[h][valid_centers[h]]:
                if v not in done_verts[h]:
                    incomplete_verts[h].append(v)
        print("Number of lines left to do: {}   {}".format(len(incomplete_verts[0]), len(incomplete_verts[1])))

        # Faster to load and map voxel weights only once
        print("Loading model info...")
        model_kwargs = dict(model_type='tikreg', story_model='story_f1', movie_model='movie_f1')
        xfm = subject[:2] + "_listening_forVL"
        storydata_wt, moviedata_wt = load_model_info(subject, xfm=xfm, **model_kwargs)

        for h in range(2):
            for vert in incomplete_verts[h]:
                print("Running job: {} {}".format(h, vert))
                _ = all_angles_xy(subject, xfm, h, vert, params, storydata_wt, moviedata_wt)
