//
// Created by peter on 2/10/23.
//
// Defines serialization types for the maps
#include "gmm_map/map.h"

namespace gmm {
    void GMMMap::save(std::ostream &stream) const {
        stream.write((char *) &num_free_clusters, sizeof(num_free_clusters));
        stream.write((char *) &num_obstacle_clusters, sizeof(num_obstacle_clusters));
        stream.write((char *) &mapParameters.track_color, sizeof(mapParameters.track_color));

        for (auto& cluster : free_clusters){
            cluster->save(stream);
        }

        for (auto& cluster : obstacle_clusters){
            cluster->save(stream);
        }

    }

    void GMMMap::load(std::istream &stream) {
        this->clear();

        stream.read((char *) &num_free_clusters, sizeof(num_free_clusters));
        stream.read((char *) &num_obstacle_clusters, sizeof(num_obstacle_clusters));
        bool track_color;
        stream.read((char *) &track_color, sizeof(mapParameters.track_color));
        if (track_color != mapParameters.track_color){
            throw std::invalid_argument(fmt::format("Loading inconsistent map binary. Change track_color to {}!", track_color));
        }

        std::cout << fmt::format("Loading binary map file with {} obstacle Gaussian, {} free Gaussian, track color: {}\n",
                                 num_obstacle_clusters, num_free_clusters, track_color);

        for (int i = 0; i < num_free_clusters; i++){
            auto new_cluster = new GMMcluster_o();
            new_cluster->load(stream);
            new_cluster->is_free = true;
            new_cluster->updateBBox(mapParameters.gau_rtree_bd);
            free_clusters.push_back(new_cluster);
            new_cluster->list_it = std::prev(free_clusters.end());
            rtree.Insert(new_cluster->BBox, new_cluster);

            if (enable_visualization){
                // Assign visualization label
                if (!new_cluster->has_label) {
                    new_cluster->has_label = true;
                    if (new_cluster->near_obs){
                        new_cluster->label = gau_free_near_obs_base + std::to_string(gau_free_idx);
                        new_cluster->bbox_label = bbox_free_base + std::to_string(gau_free_idx);
                        gau_free_idx++;
                    } else {
                        new_cluster->label = gau_free_base + std::to_string(gau_free_idx);
                        new_cluster->bbox_label = bbox_free_base + std::to_string(gau_free_idx);
                        gau_free_idx++;
                    }
                }
            }
        }

        for (int i = 0; i < num_obstacle_clusters; i++){
            auto new_cluster = new GMMcluster_c(track_color);
            new_cluster->load(stream);
            new_cluster->is_free = false;
            new_cluster->updateBBox(mapParameters.gau_rtree_bd);
            obstacle_clusters.push_back(new_cluster);
            new_cluster->list_it = std::prev(obstacle_clusters.end());
            rtree.Insert(new_cluster->BBox, new_cluster);

            if (enable_visualization){
                // Assign visualization label
                if (!new_cluster->has_label) {
                    new_cluster->has_label = true;
                    new_cluster->label = gau_obs_base + std::to_string(gau_obs_idx);
                    new_cluster->bbox_label = bbox_obs_base + std::to_string(gau_obs_idx);
                    new_cluster->color_label = gau_color_base + std::to_string(gau_obs_idx);
                    gau_obs_idx++;
                }
            }
        }

        // Update throughput statistics
        total_processed_frames++;
    }

    void GMMMap::firstClusters(std::list<GMMcluster_o *>::iterator &free_cluster_it,
                               std::list<GMMcluster_o *>::iterator &obstacle_cluster_it) {
        free_cluster_it = free_clusters.begin();
        obstacle_cluster_it = obstacle_clusters.begin();
    }

    bool GMMMap::pushClustersToVisualizationUpdateQueue(std::list<GMMcluster_o *>::iterator &free_cluster_it,
                                                        std::list<GMMcluster_o *>::iterator &obstacle_cluster_it,
                                                        const int &max_num_clusters) {
        int num_new_free_clusters = 0;
        int num_new_obs_clusters = 0;
        while (num_new_free_clusters + num_new_obs_clusters < max_num_clusters){
            if (free_cluster_it != free_clusters.end()){
                cur_visualization_update.add_free_cluster.insert(std::pair<std::string, GMMcluster_o>( (*free_cluster_it)->label,
                                                                                                       *(*free_cluster_it)));
                free_cluster_it++;
                num_new_free_clusters++;
            }

            if (obstacle_cluster_it != obstacle_clusters.end() && num_new_free_clusters + num_new_obs_clusters < max_num_clusters){
                cur_visualization_update.add_obs_cluster.insert(std::pair<std::string, GMMcluster_c>( (*obstacle_cluster_it)->label,
                                                                                                      *reinterpret_cast<GMMcluster_c*>(*obstacle_cluster_it)));
                obstacle_cluster_it++;
                num_new_obs_clusters++;
            }

            if (free_cluster_it == free_clusters.end() && obstacle_cluster_it == obstacle_clusters.end()){
                break;
            }
        }

        if (num_new_free_clusters > 0 || num_new_obs_clusters > 0){
            // Push map changes to the visualization queue
            cur_visualization_update.num_frames++;
            // Update throughput statistics
            cur_visualization_update.gmm_clustering_tp = 0;
            cur_visualization_update.gmm_fusion_tp = 0;
            cur_visualization_update.gmm_mapping_tp = 0;
            cur_visualization_update.num_free_clusters = num_free_clusters;
            cur_visualization_update.num_obs_clusters = num_obstacle_clusters;

            std::cout << fmt::format("Please wait! Loading {} obstacle Gaussian and {} free Gaussian into the visualizer!\n",
                                     num_new_obs_clusters, num_new_free_clusters);
        }

        if (free_cluster_it == free_clusters.end() && obstacle_cluster_it == obstacle_clusters.end()){
            std::cout << fmt::format("Loading completed!\n");
            return true;
        } else {
            return false;
        }
    }
}