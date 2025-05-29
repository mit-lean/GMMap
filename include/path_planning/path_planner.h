//
// Created by peter on 5/3/24.
//
// RRT Reference: https://github.com/nikhilchandak/Rapidly-Exploring-Random-Trees
//

#ifndef GMM_MAP_PATH_PLANNER_H
#define GMM_MAP_PATH_PLANNER_H
#include "gmm_map/map.h"
#include <nigh/kdtree_batch.hpp>
#include <nigh/lp_space.hpp>
#include <atomic>

namespace gmm {

    struct sampling_planner_config {
        std::string planner_name;
        bool use_occupancy;
        FP occ_free_threshold;
        FP validity_checking_resolution;
        int max_vertices;
        FP free_space_evidence;

        // Supported_planners
        std::vector<std::string> supported_planners;

        sampling_planner_config() {
            // Supported planner
            supported_planners.emplace_back("RRT");
            //supported_planners.emplace_back("RRTstar");
        }

        int getPlannerIndex(const std::string& planner){
            for (int i = 0; i < supported_planners.size(); i++){
                if (planner == supported_planners.at(i)){
                    return i;
                }
            }
            return -1;
        }
    };

    struct sampling_planner_probe {
        std::string planner_name;
        std::atomic<double> plan_duration; // in ms
        std::atomic<double> memory_usage; // in KB
        std::atomic<unsigned> num_vertices;
        std::atomic<unsigned> num_edges;
        std::atomic<double> solution_cost; // in meters
        std::atomic<bool> solution_found;

        inline sampling_planner_probe() {
            planner_name = "Unknown planner";
            reset();
        }

        inline void reset() {
            plan_duration = 0;
            memory_usage = 0;
            num_vertices = 0;
            num_edges = 0;
            solution_cost = -1;
            solution_found = false;
        }

        inline std::string toStr() {
            std::stringstream result;
            result << fmt::format("{} Planner", planner_name) << std::endl;
            result << fmt::format("Duration: {:.2f}ms", plan_duration) << std::endl;
            result << fmt::format("{} Vertices, {} Edges", num_vertices, num_edges) << std::endl;
            result << fmt::format("Memory: {:.2f}KB", memory_usage) << std::endl;
            if (solution_cost == -1){
                result << "No feasible solution!" << std::endl;
            } else {
                result << fmt::format("Solution cost: {:.2f}m", solution_cost) << std::endl;
            }
            return result.str();
        }
    };

    template<typename T>
    class TreeNode
    {
    public:
        TreeNode() {
            clear();
        }

        TreeNode(const T& value, TreeNode<T>* parent = nullptr) {
            value_ = value;
            parent_ = parent;
        }

        TreeNode<T>* insertChild(const T& value) {
            children_.emplace_back(value, this);
            children_ptr_.emplace_back(&children_.back());
            return &(children_.back());
        }

        void clear() {
            parent_ = nullptr;
            children_.clear();
            children_ptr_.clear();
        }

        std::list<TreeNode<T>*> backTraceToRoot() {
            std::list<TreeNode<T>*> path;
            path.emplace_back(this);

            auto cur_node = this;
            while (cur_node->parent_ != nullptr){
                path.emplace_back(cur_node->parent_);
                cur_node = cur_node->parent_;
            }
            return path;
        }

        std::list<TreeNode<T>*> getAllChildNodes() {
            std::list<TreeNode<T>*> result;
            for (auto& child : children_){
                result.push_back(&child);
                auto child_result = child.getAllChildNodes();
                result.splice(result.end(), child_result);
            }
            return result;
        }

        T value_;
        TreeNode<T>* parent_;
        std::list<TreeNode<T>> children_;
        std::list<TreeNode<T>*> children_ptr_;
    };


    class GMMapSamplingBasedPlanner
    {
    public:
        GMMapSamplingBasedPlanner(const GMMMap* map_ptr, const sampling_planner_config& config, sampling_planner_probe* probe = nullptr);

        void updatePlannerConfig(const sampling_planner_config& config);

        void updatePlanner(int i);

        void updateMaxVertices(int max_vertices);

        bool plan(const V& start_coord, const V& goal_coord, bool relative_coordinate = false);

        void clearPlanner();

        void exportGraph(std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector2i>& edges) const;

        void exportSolutionPath(std::vector<Eigen::Vector3d>& points, std::vector<Eigen::Vector2i>& edges) const;

        sampling_planner_probe* getProbePtr() const;

        bool haveSolutionPath() const;
        std::list<V> getSolutionPath() const;
        float getSolutionCost() const;

    private:
        // Steer the point from start towards t in delta increment
        V steerPoint(const V& start, const V& t, FP delta, FP eps = 0.001);

        void RRT(const V &start_coord, const V &goal_coord);
        bool shouldTerminate() const;
        std::vector<V> randomSampleInBBox(const Rect& bbox, int num_samples = 1) const;
        void checkLineSegmentCollision(const V& start, const V& end, V& nearest) const;
        bool checkLineSegmentCollision(const V& start, const V& end) const;
        std::vector<bool> isStatesValid(const std::vector<V>& pts) const;
        bool isStateValid(const V& pt) const;

        const GMMMap* map_;
        TreeNode<V> planner_tree_root_;
        TreeNode<V>* solution_node_ptr_; // Points to the next node that connects the cur_goal_location
        int num_of_tree_nodes_;

        // Tracks the current cost of solution
        V cur_goal_location_; // Caches the current goal location
        FP cur_solution_cost_; // Contains the cost of the path
        int num_of_solutions_; // Number of solutions found during the search

        sampling_planner_config config_;
        sampling_planner_probe* probe_;

        // Define KD-Tree structure
        // See: https://github.com/UNC-Robotics/nigh/blob/master/demo/euclidean_demo.cpp
        struct TreeNodeKey {
            const V& operator() (const TreeNode<V>* node) const {
                return node->value_;
            }
        };

        using Space = unc::robotics::nigh::LPSpace<FP, 3, 2>;
        using kdtree = unc::robotics::nigh::Nigh<TreeNode<V>*,Space, TreeNodeKey,
                        unc::robotics::nigh::NoThreadSafety,unc::robotics::nigh::KDTreeBatch<>>;
        kdtree nn;
    };
}

#endif //GMM_MAP_PATH_PLANNER_H
