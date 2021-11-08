import numpy as np
import cortex
import cortex.polyutils
import os
import copy
from scipy.stats import linregress

from vl_interface.utils import geodesic_path, anti_geodesic_path, load_pcs
from vl_interface import CC_INTERFACE

import cottoncandy as cc
cci = cc.get_interface(CC_INTERFACE, verbose=False)


class BrainLine(object):
    '''
    Object that is line drawn across the surface of cortex
    '''
    def __init__(self, subject, xfm, hem, param_set, verts=None):
        self.subject = subject
        self.xfm = xfm
        self.hem = hem  #0==left, 1==right
        self.param_set = param_set['name']
        #Can initialize with verts or assign them later through self.draw_line
        self.verts = verts
        self.ROI_verts = None
        self.ap_locs = None
        self.subdiv_verts = None
        self.subdiv_ap = None
        self.native_surf = True #Changess to false if subsurf is used for verts/ROI_verts
        self.sem_vectors = None
        self.v_sem_vector = None
        self.l_sem_vector = None
        self.slope_diffs = None
        self.wt_corr = None
        self.r_squared = None

    def check_surf(self, surf):
        # Check if surface has subsurface properties
        if hasattr(surf, 'subsurface_vertex_map'):
            self.native_surf = False
        else:
            self.native_surf = True

    def draw_line(self, a, b, border_pts, surf, m=1.0):
        '''Draw a line from with starting vertex a, going through vertex b,
        going away from point a until hitting one of the other border_pts on the
        subsurf.
        The value of m controls how smooth the line is (Reverse Euler step length)
        '''
        #surf = self.get_surf()
        # Draw line from vertex a to vertex b
        half_line = geodesic_path(a, b, surf, m=m)
        # Draw line from vertex b, going away from vertex a, until you leave "pts"
        anti_line = anti_geodesic_path(b, a, border_pts, surf, m=m)
        # Combine lines together
        if (anti_line is not None) and (half_line is not None):
            self.verts = np.hstack((half_line, anti_line[1:]))
        self.check_surf(surf)

    def get_ap_locs(self, surf, dist, smooth_factor=2.0):
        '''Find points within dist mm around the line defined by self.verts
        Also find the anterior-posterior locations of each vertex in this ROI
        '''
        assert self.verts is not None
        # First find points within dist of the line
        all_dists = np.array([surf.geodesic_distance([v]) for v in self.verts])
        min_dists = np.min(all_dists, axis=0)
        self.ROI_verts = np.where(min_dists < dist)[0]
        # Then find distances from all points in ROI to all points on the line
        ROI_dists_subset = np.array([[all_dists[v][p] for p in self.ROI_verts] for v in range(len(self.verts))])
        # Get distances between the points on the line
        line_interdists = np.zeros(len(self.verts)-1)
        for num,v in enumerate(self.verts[:-1]):
            line_interdists[num] = np.linalg.norm(surf.pts[v] - surf.pts[self.verts[num+1]])
        # And cumulative distance from the start of the line
        ROI_vert_ap_locs = np.hstack([[0], np.cumsum(line_interdists)])
        # Find "weights" of how far each ROI vertex is from each point on the line
        normsum = lambda v: v / v.sum(0)
        ROI_wts = normsum(np.exp(-ROI_dists_subset / smooth_factor))
        # Anterior-posterior location along the line is the dot product of weights 
        # and the line locations
        self.ap_locs = np.dot(ROI_vert_ap_locs, ROI_wts)
        self.check_surf(surf)
        if not self.native_surf:
            self.subsurf_to_native(surf)

    def subdivide_center_vert(self, dist, ap_params, vert):
        '''Special case of subdividing where we want just one subdivision, 
        and it is centered on the given vert
        '''
        # Where is the center vert in the list
        idx = np.where(self.ROI_verts==vert)[0][0]
        ap_mid = self.ap_locs[idx]
        ap_min = ap_mid - ap_params[0]/2
        ap_max = ap_mid + ap_params[0]/2

        # Make the single subdivision
        self.subdiv_verts = []
        self.subdiv_ap = []
        for v,ap in zip(self.ROI_verts, self.ap_locs):
            if ap >= ap_min and ap <= ap_max:
                self.subdiv_verts.append(v)
                self.subdiv_ap.append(ap)

    def subsurf_to_native(self, surf):
        if self.native_surf:
            print("Already in native surface space, why are you doing this???")
        else:
            try:
                self.verts = surf.subsurface_vertex_inverse[self.verts]
                self.ROI_verts = surf.subsurface_vertex_inverse[self.ROI_verts]
                self.native_surf = True
            except:
                Exception("Incorrect subsurface provided... or surface operations not complete yet...")

    def semantic_analysis(self, dist, ap_params, storydata_wt, moviedata_wt):
        '''Use model weights for each vertex in the ROI around the line to see how the 
        visual vs. linguistic representations are changing with AP location
        
        Project onto visual OR language vector, then select which is stronger
        '''
        assert self.subdiv_ap is not None
        
        subset_storydata_wt = storydata_wt[self.hem][:,self.subdiv_verts]
        subset_moviedata_wt = moviedata_wt[self.hem][:,self.subdiv_verts]

        avg_swts = np.mean(subset_storydata_wt, axis=1)
        avg_mwts = np.mean(subset_moviedata_wt, axis=1)
        
        self.l_sem_vector = avg_swts / np.linalg.norm(avg_swts, 2)
        self.v_sem_vector = avg_mwts / np.linalg.norm(avg_mwts, 2)

        v_story_ROI = np.dot(self.v_sem_vector, subset_storydata_wt)
        v_movie_ROI = np.dot(self.v_sem_vector, subset_moviedata_wt)

        v_story_fit = linregress(self.subdiv_ap, v_story_ROI)
        v_movie_fit = linregress(self.subdiv_ap, v_movie_ROI)

        l_story_ROI = np.dot(self.l_sem_vector, subset_storydata_wt)
        l_movie_ROI = np.dot(self.l_sem_vector, subset_moviedata_wt)

        l_story_fit = linregress(self.subdiv_ap, l_story_ROI)
        l_movie_fit = linregress(self.subdiv_ap, l_movie_ROI)
        
        # Find metric for vision projections
        v_slope_story = v_story_fit[0]
        v_slope_movie = v_movie_fit[0]
        v_int_story = v_story_fit[1]
        v_int_movie = v_movie_fit[1]

        v_div = np.array([v_slope_movie/v_slope_story, v_slope_story/v_slope_movie])
        v_div_metric = v_div[np.argmin(np.abs(v_div))]
        v_slope_mag = np.mean([np.abs(v_slope_story), np.abs(v_slope_movie)])

        # Did some basic algebra to find this
        v_ap_split = (v_int_story - v_int_movie) / (v_slope_movie - v_slope_story)
        v_cross_indicator = int(np.logical_and(max(self.subdiv_ap) > v_ap_split,
                                               min(self.subdiv_ap) < v_ap_split))
        v_pos_indicator = int(v_story_ROI.mean() * v_movie_ROI.mean() > 0)

        v_met = -1*v_div_metric*v_slope_mag*v_cross_indicator*v_pos_indicator
        
        # Now repeat for language projections
        l_slope_story = l_story_fit[0]
        l_slope_movie = l_movie_fit[0]
        l_int_story = l_story_fit[1]
        l_int_movie = l_movie_fit[1]

        l_div = np.array([l_slope_movie/l_slope_story, l_slope_story/l_slope_movie])
        l_div_metric = l_div[np.argmin(np.abs(l_div))]
        l_slope_mag = np.mean([np.abs(l_slope_story), np.abs(l_slope_movie)])

        # Did some basic algebra to find this
        l_ap_split = (l_int_story - l_int_movie) / (l_slope_movie - l_slope_story)
        l_cross_indicator = int(np.logical_and(max(self.subdiv_ap) > l_ap_split,
                                               min(self.subdiv_ap) < l_ap_split))
        l_pos_indicator = int(l_story_ROI.mean() * l_movie_ROI.mean() > 0)

        l_met = -1*l_div_metric*l_slope_mag*l_cross_indicator*l_pos_indicator
        
        # Find max value across modalities
        self.slope_diffs = max([v_met, l_met])
        if v_met >= l_met:
            self.v_or_l = 'v'
            do_flip = True if v_slope_story < 0 else False 

            if do_flip:
                # "anterior" verts are actually those "posterior" before the flip
                v_ant_verts = np.array(self.subdiv_verts)[self.subdiv_ap < v_ap_split]
                # and vice versa
                v_pos_verts = np.array(self.subdiv_verts)[self.subdiv_ap > v_ap_split]
            else:
                v_ant_verts = np.array(self.subdiv_verts)[self.subdiv_ap > v_ap_split]
                v_pos_verts = np.array(self.subdiv_verts)[self.subdiv_ap < v_ap_split]
            v_avg_swts = np.mean(storydata_wt[self.hem][:, v_ant_verts], axis=1)
            v_avg_mwts = np.mean(moviedata_wt[self.hem][:, v_pos_verts], axis=1)
            v_ant_sem = v_avg_swts / np.linalg.norm(v_avg_swts, 2)
            v_pos_sem = v_avg_mwts / np.linalg.norm(v_avg_mwts, 2)

            self.wt_corr = np.corrcoef(v_ant_sem, v_pos_sem)[0][1]
        else:
            self.v_or_l = 'l'
            do_flip = True if l_slope_story < 0 else False 

            if do_flip:
                # "anterior" verts are actually those "posterior" before the flip
                l_ant_verts = np.array(self.subdiv_verts)[self.subdiv_ap < l_ap_split]
                # and vice versa
                l_pos_verts = np.array(self.subdiv_verts)[self.subdiv_ap > l_ap_split]
            else:
                l_ant_verts = np.array(self.subdiv_verts)[self.subdiv_ap > l_ap_split]
                l_pos_verts = np.array(self.subdiv_verts)[self.subdiv_ap < l_ap_split]
            l_avg_swts = np.mean(storydata_wt[self.hem][:, l_ant_verts], axis=1)
            l_avg_mwts = np.mean(moviedata_wt[self.hem][:, l_pos_verts], axis=1)
            l_ant_sem = l_avg_swts / np.linalg.norm(l_avg_swts, 2)
            l_pos_sem = l_avg_mwts / np.linalg.norm(l_avg_mwts, 2)

            self.wt_corr = np.corrcoef(l_ant_sem, l_pos_sem)[0][1]
        
        # Now reorder line.verts and line.subdiv_ap
        if do_flip:
            subdiv_ap = self.subdiv_ap
            ap_locs = self.ap_locs
            subdiv_mid = (max(subdiv_ap) + min(subdiv_ap)) / 2

            vert_aps = []
            for v in self.verts:
                index = np.where(self.ROI_verts==v)[0][0]
                vert_aps.append(self.ap_locs[index])
            diff_from_mids = np.abs(subdiv_mid - vert_aps)
            min_idx = np.argmin(diff_from_mids)

            # This is the very middle of the ROI
            center = vert_aps[min_idx]

            ap_diffs = center - subdiv_ap
            ap_new = center + ap_diffs
            self.subdiv_ap = ap_new

            ap_diffs_all = center - ap_locs
            ap_new_all = center + ap_diffs_all
            self.ap_locs = ap_new_all

            self.verts = self.verts[::-1]
