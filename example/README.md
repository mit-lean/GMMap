# Adopting GMMap to Your Applications #
Hyper-parameters of GMMap are defined in `[dataset_name].json` files. Some examples can be found in [indoor](indoor) and [outdoor](outdoor) directories. Note that some parameters were used internally for debugging and thus are not active. Please refer to the following list of active parameters in each section.

## Active Parameters
### cpu_computer_parameters
1. **num_threads**: Maximum number of CPU cores used in SPGF. Each core will execute Scanline Segmentation (SS) for a row of pixels in depth image.

### gpu_compute_parameters
For CUDA implementation, Scanline Segmentation (SS) in SPGF is accelerated via CUDA. Everything else is ran on CPU. Each CUDA thread executes SS for a row in the depth image.
1. **num_blocks**: Number of thread blocks that are active.
2. **threads_per_block**: Number of threads per block.
3. **max_segments_per_line**: Maximum number of line segments that can be stored in the buffer per row. Segments that are created after the buffer is full are pruned.

### gmm_clustering_parameters
This section contains parameters for SPGF* (referred to as `spgf_extended` in this codebase). For more details about these parameters, please refer to the ICRA paper [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9811682).

#### scanline_segmentation
The following parameters are used in Scanline Segmentation (SS).
1. **adaptive_thresh_scale**: The reciprocal of parameter `b` in Equation 3. Note that the parameter `a` is set to six.
2. **noise_floor**: The minimum threshold `n_min` along the z-axis (in meters) below which the current depth pixel can correspond to the same line segment (in SS) or same plane (in Segment Fusion).
3. **occ_x_t**: The threshold `t_occ` that defines the maximum number of pixels along the x-axis that a previous line segment could be occluded before merging with the current pixel.
4. **noise_thresh**: The maximum threshold along the z-axis (in meters) below which the current depth pixel can correspond to the same line segment (in SS) or same plane (in Segment Fusion).
5. **ncheck_t**: The threshold `t_fit` that defines the number of pixels in a previous line segment after which the parameters of the line can be accurately estimated.
6. **line_t**: The parameter that determines the maximum difference (in meters) along the x-axis for the current pixel to correspond to the line segment.
7. **max_incomplete_clusters**: The parameter `beta` that determines the maximum number of previous segments that can be considered for fusing with the current depth pixel.

#### segment_fusion
The following parameter is used in Segment Fusion (SF).
1. **angle_t**: Parameter `t_cos` in the ICRA paper. Determines the threshold above which line segments are sufficiently parallel so that they belong to the same plane.

#### spurious_gaussian_purge
These parameters are used to prune small Gaussians that are likely represent noise in the depth measurements in order to enhance the compactness of the map. However, setting these values too large will eliminate intricate details of the environment that you may wish to preserve. 
1. **sparse_t**: Line segments after SS that contains less than **sparse_t** pixels are pruned.
2. **num_line_t**: Gaussians after SF that are comprised of less than **num_line_t** line segments are pruned.
3. **num_pixels_t**: Gaussians after SF that contains less than **num_pixels_t** pixels are pruned.

#### free_gaussian_generation
Please leave `free_space_dist_scale = 1` unchanged. This parameter was used for some internal debugging purposes.

### gmm_fusion_parameters
The following parameters are used for globally-consistent Gaussian fusion in GMMap. Please refer to the TRO paper [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10379145) for details. To avoid the squared root during Hellinger distance computation (and thus the resulting distance is squared), all Hellinger distance thresholds are squared (`hell_thresh_squared_*`) as well.
1. **hell_thresh_squard_free**: The parameter `alpha_h,free` squared for determining similarity of two free Gaussians.
2. **hell_thresh_squard_obs_scale**: The parameter `alpha_h,occ` is computed as `(alpha_h,occ)^2 = (alpha_h,free)^2 * hell_thresh_squard_obs_scale`. Since free Gaussians are created from basis derived from occupied Gaussians, the scale `hell_thresh_squard_obs_scale` captures this correlation during fusion.
3. **hell_thresh_squard_oversized_gau**: A conservative threshold for fusing large Gaussians (identified using `fusion_max_scale`). A large Gaussian may not represent the environment well. Thus, this threshold is typically very small to ensure that two large Gaussians are fused only when they are extremely similar.
4. **hell_thresh_squard_min**: A minimum hellinger distance threshold (squared) below which Gaussians are fused. This is used to prevent special cases where the final threshold (`s_r * alpha_h` in Line 17 of Algorithm 3) is near zero, which prevents fusion from happening and cause the map size to increase significantly,
5. **min_gau_len**: The partition plane distance `d_0` during the creation of subregions for free Gaussians in SPGF*. See Equation 20a.
6. **frame_max_scale**: Determines the maximum distance (`min_gau_len * frame_max_scale`) between two partitioning planes of each subregion for free Gaussians in SPGF*. Setting the maximum distance can avoid the construction of large Gaussians that may not represent the free regions well.
7. **fusion_max_scale**: Determines the maximum side length of the bounding box (`min_gau_len * fusion_max_scale`) for identifying large Gaussians so that a more conservative threshold of `hell_thresh_squard_oversized_gau` is applied during fusion.
8. **gau_fusion_bd**: Defines the extent of the bounding box used to enclose each Gaussian (*i.e.*, number of Mahalanobis distance from the Gaussian mean) for determining whether two Gaussians intersect (Lines 7 & 10 in Algorithm 3).
9. **gau_rtree_bd**: Defines the extent of the bounding box used to enclose each Gaussian (*i.e.*, number of Mahalanobis distance from the Gaussian mean) in the R-Tree .
10. **track_color**: Determine whether we want to fuse the RGB value corresponding to each depth pixel during Gaussian construction. Set to `false` if we only need an occupancy map.
11. **track_intensity**: Determine whether we want to fuse the color intensity value corresponding to each depth pixel during Gaussian construction. Set to `false` if we only need an occupancy map.

### occupancy_inference_parameters
The following parameters are used for computing the occupancy probability during Gaussian Mixture Regression (GMR). Please refer to the TRO paper [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10379145) for details.
1. **unexplored_evidence**: Weight `pi_0` of the unexplored prior. A larger evidence will require more depth measurements to push the occupancy probability from 0.5 towards 0 (free) or occupied (1).
2. **unexplored_variance**: Parameter `sigma^2_0` in Equation 5. 
3. **min_num_neighbor_clusters**: Not used.

## Tips for Tuning Parameters
We strongly recommend tuning the parameters for Gaussians generated from each image first (*i.e.*, SPGF* only) before fusion across images. This can be achieved by the following steps:
1. Initialize your settings using the ones from [tum.json](indoor/tum.json). Make sure to change the camera parameters and dataset locations accordingly. 
2. Compile the GUI executable `GMMMapVisualization`. Make sure to uncheck the `Fuse GMM across frames` box from the `Starting Settings` panel before execution. In this mode, the Gaussians fusion across images are disabled (*i.e.*, Gaussians generated from SPGF* are plotted). Tune the **scanline_segmentation** and **segment_fusion** parameters to make sure the resulting Gaussians look desirable.
3. Rerun the GUI executable `GMMMapVisualization` again while `Fuse GMM across frames` is checked. In this mode, the fused Gaussians across images are illustrated. Tune the **gmm_fusion_parameters** to make sure the resulting Gaussians look desirable.

## Commonly-Asked Questions
1. **Free Gaussians are not showing up in the visualizer?**: Make sure to check both `Show GMM Free` and `Show GMM Free (Near Obs)` boxes in the GUI.
2. **Visualizer crashes / segmentation faults?**: Make sure that the GMMap construction is not too fast relative to the visualizer. You can increase the `Update Interval` and / or `Visualization Latency`.
3. **Gaussians are not modelling small objects?**: Firstly, we recommend reducing the Gaussian pruning thresholds **num_line_t** and **num_pixels_t** so that Gaussians representing small objects (and noise) are preserved. Secondly, we recommend lowering the one of the (or both) fusion thresholds **hell_thresh_squard_free** and **hell_thresh_squard_obs_scale** so that only Gaussians that are extremely similar are fused across images.