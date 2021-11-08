import cortex
import numpy as np
from scipy.spatial import Delaunay
import networkx as nx

from vl_interface import CC_INTERFACE

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)


def flatpx2barycentric(subject, sulc_pts):
    """pts is 2xN array of pt locations in flatmap SVG pixel space.
    """
    roi = cortex.db.get_overlay(subject)
    pts, polys = cortex.db.get_surf(subject, "flat", merge=True, nudge=True)

    norm_pts = np.zeros((pts.shape[0], 2))
    for idx in range(2):
        norm_pts[:,idx] = (pts[:,idx] + np.abs(pts[:,idx].min()))
        norm_pts[:,idx] *= (1./norm_pts[:,idx].max())
    norm_pts[:,1] = 1-norm_pts[:,1]

    flat = norm_pts * roi.svgshape
    valid = np.unique(polys)

    ## Get barycentric coordinates
    dl = Delaunay(flat[valid,:2])
    simps = dl.find_simplex(sulc_pts)
    missing = simps == -1
    tfms = dl.transform[simps]
    l1, l2 = (tfms[:,:2].transpose(1,2,0) * (sulc_pts - tfms[:,2]).T).sum(1)
    l3 = 1 - l1 - l2

    ll = np.vstack([l1, l2, l3])
    ll[:,missing] = 0

    return np.arange(len(flat))[valid][dl.vertices[simps]], ll

def closest_verts(verts, bcs):
    return verts[range(len(verts)), bcs.argmax(0)]

def vts2line(surf, vts):
    path = [vts[0]] # initialize path with first vertex

    for frm, to in zip(vts, vts[1:]):
        p = nx.shortest_path(surf.graph, frm, to) # includes start and end
        path += p[1:] # snip off start from each

    return path

def get_occ_lines(subject, geo_path=False):
    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
    numl = surfs[0].pts.shape[0]

    overlays = cortex.db.get_overlay(subject)
    occ_splines = overlays.sulci.shapes['occ_lobe'].splines
    
    occ_pts = [o.vertices for o in occ_splines]
    occ_bary = [flatpx2barycentric(subject, occ) for occ in occ_pts]
    occ_closest = [closest_verts(*occ_b) for occ_b in occ_bary]
    if geo_path:
        occ_lines = []
    else:
        occ_lines = [np.array(vts2line(surfs[h],occ_closest[h]-h*numl)) for h in range(2)]

    return occ_lines

def select_occ_band(subject, geo_path=False, distance=50):
    occ_lines = get_occ_lines(subject, geo_path=geo_path)

    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]

    occ_dists = [surfs[h].geodesic_distance(occ_lines[h]) for h in range(2)]
    occ_bands = [np.where(dists <= distance)[0] for dists in occ_dists]

    return occ_lines, occ_bands
    
def plot_occ_band(subject, geo_path=False, distance=50, save=True, outfile=None):
    occ_lines, occ_bands = select_occ_band(subject, geo_path=geo_path, distance=distance)

    surfs = [cortex.polyutils.Surface(*d) for d in cortex.db.get_surf(subject, "fiducial")]
    numl = surfs[0].pts.shape[0]
    numr = surfs[1].pts.shape[0]

    band_verts = np.empty(numl+numr)
    band_verts[:] = np.nan

    for h in range(2):
        band_verts[occ_bands[h]+h*numl] = 1
        band_verts[occ_lines[h]+h*numl] = 2

    vol = cortex.Vertex(band_verts, subject, vmin=1, vmax=2, cmap='inferno')
    
    if save and outfile is not None:
        h = cortex.quickflat.make_png(outfile.format(subject, distance), vol,
                                      with_curvature=True, with_labels=False,
                                      with_colorbar=False, linewidth=3)
    else:
        h = cortex.quickshow(vol, with_curvature=True, with_labels=False,
                             with_colorbar=False, linewidth=3)

    return h

def save_band(subject, geo_path=False, distance=50):
    line_file = "occ_line/{}_h{}.ccd"
    band_file = "occ_band/{}_{}mm_h{}.ccd"

    occ_lines, occ_bands = select_occ_band(subject=subject, geo_path=geo_path, distance=distance)

    for h in range(2):
        cci.upload_raw_array(line_file.format(subject, h), occ_lines[h])
        cci.upload_raw_array(band_file.format(subject, distance, h), occ_bands[h])

    print("Saving complete for {}, {}mm".format(subject, distance))

def load_band(subject, hem, distance=50):
    band_file = "occ_band/{}_{}mm_h{}.ccd"
    occ_band = cci.download_raw_array(band_file.format(subject, distance, hem))

    return occ_band
