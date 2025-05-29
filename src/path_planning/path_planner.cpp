//
// Created by peterli on 5/6/24.
//

#include <path_planning/path_planner.h>
#include <chrono>
#include <unordered_map>

namespace gmm {

    GMMapSamplingBasedPlanner::GMMapSamplingBasedPlanner(const GMMMap *map_ptr, const sampling_planner_config &config,
                                                         sampling_planner_probe *probe) {
        map_ = map_ptr;
        probe_ = probe;
        if (probe != nullptr) {
            probe_->reset();
            probe_->planner_name = config_.planner_name;
        }

        // Initialize planner settings
        updatePlannerConfig(config);

        // Setup planner tree
        clearPlanner();
    }

    void GMMapSamplingBasedPlanner::updatePlannerConfig(const gmm::sampling_planner_config &config) {
        config_ = config;
        if (probe_ != nullptr) {
            probe_->planner_name = config_.planner_name;
            probe_->reset();
        }

        // Reset planner
        clearPlanner();
    }

    void GMMapSamplingBasedPlanner::updatePlanner(int i) {
        if (i > config_.supported_planners.size() - 1) {
            return;
        }

        auto selected_planner = config_.supported_planners.at(i);
        if (config_.planner_name == selected_planner) {
            return;
        } else {
            config_.planner_name = selected_planner;
            updatePlannerConfig(config_);
        }
    }

    bool GMMapSamplingBasedPlanner::plan(const gmm::V &start_coord, const gmm::V &goal_coord, bool relative_coordinate) {
        // Actual path-planning class
        V actual_start_coord, actual_goal_coord;
        if (relative_coordinate) {
            actual_start_coord = map_->relativeToAbsoluteCoordinate(
                    start_coord(0), start_coord(1), start_coord(2));
            actual_goal_coord = map_->relativeToAbsoluteCoordinate(
                    goal_coord(0), goal_coord(1), goal_coord(2));

        } else {
            actual_start_coord = start_coord;
            actual_goal_coord = goal_coord;
        }

        auto tic = std::chrono::steady_clock::now();

        RRT(actual_start_coord, actual_goal_coord);

        auto toc = std::chrono::steady_clock::now();

        // Update probe
        if (probe_ != nullptr) {

            probe_->plan_duration =
                    (double) std::chrono::duration_cast<std::chrono::microseconds>(toc - tic).count() / 1000.0; // in ms
            probe_->solution_found = haveSolutionPath();
            probe_->num_vertices = num_of_tree_nodes_;
            probe_->num_edges = num_of_tree_nodes_ / 2;

            // Assuming each vertex contains a 3D coordinate. Each edge is stored with the source vertex and contains the id to target vertex.
            probe_->memory_usage =
                    (probe_->num_vertices * gmm::V::SizeAtCompileTime + probe_->num_edges) * sizeof(gmm::FP) /
                    1024.0; // in KB
            probe_->solution_cost = getSolutionCost(); // in meters
        }
        return haveSolutionPath();
    }

    void GMMapSamplingBasedPlanner::exportSolutionPath(std::vector<Eigen::Vector3d> &points,
                                                       std::vector<Eigen::Vector2i> &edges) const {
        points.clear();
        edges.clear();

        if (!haveSolutionPath()) {
            return;
        }

        std::cout << fmt::format("Number of solutions found: {}", num_of_solutions_) << std::endl;

        // Get the best solution from all available ones
        auto path = getSolutionPath();

        // Obtain vertices for solutions
        std::cout << fmt::format("Exporting solution path with {} states ...", path.size()) << std::endl;

        points.reserve(path.size());
        edges.reserve(path.size() - 1);

        double cost = 0;
        int path_idx = 0;
        auto path_itr = path.begin();
        while (path_itr != path.end()) {
            points.emplace_back((*path_itr).cast<double>());
            if (path_idx > 0) {
                edges.emplace_back(path_idx - 1, path_idx);
                cost += (points.at(points.size() - 1) - points.at(points.size() - 2)).norm();
            }
            path_idx++;
            path_itr++;
        }

        std::cout << fmt::format("Solution cost (path length) is {:.2f}m", cost) << std::endl;
    }

    void GMMapSamplingBasedPlanner::exportGraph(std::vector<Eigen::Vector3d> &points,
                                                std::vector<Eigen::Vector2i> &edges) const {
        // Added by Peter to print the underlying graph
        auto tree_root = planner_tree_root_;
        auto vertices = tree_root.getAllChildNodes();
        vertices.push_front(&tree_root);

        points.clear();
        edges.clear();

        points.reserve(vertices.size());
        edges.reserve(vertices.size() / 2); // Only draw one edge in the undirected graph

        std::unordered_map<TreeNode<V>*, int> node_to_idx_map;

        for (auto& vertex : vertices){
            // Find node index
            int node_idx;
            auto it = node_to_idx_map.find(vertex);
            if (it == node_to_idx_map.end()){
                node_to_idx_map.insert(std::make_pair(vertex, points.size()));
                node_idx = points.size();
                points.emplace_back(vertex->value_.cast<double>());
                //::cout << "Showing points" << std::endl;
                //std::cout << points.back() << std::endl;
            } else {
                node_idx = it->second;
            }

            for (auto& child_ptr : vertex->children_ptr_) {
                // Find children
                int child_idx;
                it = node_to_idx_map.find(child_ptr);
                if (it == node_to_idx_map.end()){
                    node_to_idx_map.insert(std::make_pair(child_ptr, points.size()));
                    child_idx = points.size();
                    points.emplace_back(child_ptr->value_.cast<double>());
                } else {
                    child_idx = it->second;
                }
                edges.emplace_back(node_idx, child_idx);

                /*
                if (!points.at(node_idx).allFinite() || !points.at(child_idx).allFinite()){
                    throw std::invalid_argument("One of the vertices in the graph is not finite!");
                }

                if (checkLineSegmentCollision(points.at(node_idx).cast<FP>(), points.at(child_idx).cast<FP>())){
                    throw std::invalid_argument("Line segment between two vertices collide!");
                }
                */
            }
        }

        // Connect goal to the correct leaf
        if (haveSolutionPath()){
            points.emplace_back(cur_goal_location_.cast<double>());
            auto it = node_to_idx_map.find(solution_node_ptr_);
            edges.emplace_back(it->second, points.size()-1);
        }
    }

    void GMMapSamplingBasedPlanner::clearPlanner() {
        // Setup planner tree
        planner_tree_root_.clear();
        num_of_tree_nodes_ = 0;
        solution_node_ptr_ = nullptr;
        num_of_solutions_ = 0;
        nn.clear();
    }

    bool GMMapSamplingBasedPlanner::haveSolutionPath() const {
        return solution_node_ptr_ != nullptr;
    }

    std::list<V> GMMapSamplingBasedPlanner::getSolutionPath() const {
        std::list<V> result;
        if (haveSolutionPath()){
            auto root = planner_tree_root_;
            auto solution = solution_node_ptr_->backTraceToRoot();
            solution.reverse();
            for (auto& v : solution){
                result.emplace_back(v->value_);
            }
        }
        result.emplace_back(cur_goal_location_);
        return result;
    }

    float GMMapSamplingBasedPlanner::getSolutionCost() const {
        if (solution_node_ptr_ == nullptr) {
            return -1;
        }

        // Get the best solution from all available ones
        auto path = getSolutionPath();
        float cost = 0;

        auto pre_p = path.front();
        for (auto& p : path) {
            if (p != pre_p) {
                cost += (p - pre_p).norm();
            }
            pre_p = p;
        }
        return cost;
    }

    std::vector<bool> GMMapSamplingBasedPlanner::isStatesValid(const std::vector<V>& pts) const {
        std::vector<bool> results;
        results.reserve(pts.size());
        // TODO: Replace by batch sampling in GMMap hardware
        for (auto& pt : pts) {
            FP occ = map_->computeOccupancy(pt, config_.free_space_evidence);
            bool isValid;
            if (config_.use_occupancy) {
                isValid =  occ < config_.occ_free_threshold;
            } else {
                isValid =  occ < 0.5f;
            }
            results.push_back(isValid);
        }
        return results;
    }

    bool GMMapSamplingBasedPlanner::isStateValid(const gmm::V &pt) const {
        FP occ = map_->computeOccupancy(pt, config_.free_space_evidence);
        bool isValid;
        if (config_.use_occupancy) {
            isValid =  occ < config_.occ_free_threshold;
        } else {
            isValid =  occ < 0.5f;
        }
        return isValid;
    }

    std::vector<V> GMMapSamplingBasedPlanner::randomSampleInBBox(const Rect& bbox, int num_samples) const {
        std::vector<V> results;
        results.reserve(num_samples);
        for (int i = 0; i < num_samples; i++){
            results.emplace_back(bbox.sample());
        }
        return results;
    }

    void GMMapSamplingBasedPlanner::checkLineSegmentCollision(const V& start, const V& end, V& nearest) const {
        int num_of_points = (int) std::ceil((end - start).norm() / config_.validity_checking_resolution) + 1;

        // Generate points along the line
        auto unit_vector = config_.validity_checking_resolution * (end - start).normalized();
        for (int i = 0; i < num_of_points; i++){
            V pt;
            if (i == num_of_points - 1) {
                pt = end;
            } else {
                pt = start + i * unit_vector;
            }
            auto point_valid = isStateValid(pt);
            if (point_valid){
                nearest = pt;
            } else {
                if (i == 0){
                    throw std::invalid_argument(fmt::format("Initial state at the start of line segment [{:.2f}, {:.2f}, {:.2f}] cannot be invalid!",
                                                            pt(0), pt(1), pt(2)));
                }
                return;
            }
        }
    }

    bool GMMapSamplingBasedPlanner::checkLineSegmentCollision(const V &start, const V &end) const {
        V nearest;
        checkLineSegmentCollision(start, end, nearest);
        return (nearest != end);
    }

    sampling_planner_probe *GMMapSamplingBasedPlanner::getProbePtr() const {
        return probe_;
    }

    bool GMMapSamplingBasedPlanner::shouldTerminate() const {
        // Current termination condition is defined based on the number of vertices
        if (num_of_tree_nodes_ > config_.max_vertices) {
            return true;
        } else {
            return false;
        }
    }

    void GMMapSamplingBasedPlanner::updateMaxVertices(int max_vertices) {
        config_.max_vertices = max_vertices;
    }

    V GMMapSamplingBasedPlanner::steerPoint(const V& start, const V& t, FP delta, FP eps) {
        FP distance = (t - start).norm();
        if( (distance - delta) <= eps)
            return t ;
        else {
            V unit_vector = (t - start).normalized();
            return start + unit_vector * delta;
        }
    }

    void GMMapSamplingBasedPlanner::RRT(const V &start_coord, const V &goal_coord) {
        clearPlanner();
        int iteration_idx = 0;
        int solution_check_interval = 100;

        cur_goal_location_ = goal_coord;
        V sample;

        Rect env_bbox;
        map_->getEnvBounds(env_bbox);

        // Inserting start node into the tree
        planner_tree_root_.value_ = start_coord;
        nn.insert(&planner_tree_root_);
        num_of_tree_nodes_++;

        while(!shouldTerminate()) {
            //std::cout << fmt::format("Planning iteration {}", iteration_idx) << std::endl;
            if (iteration_idx % solution_check_interval == 0){
                sample = cur_goal_location_;
                //std::cout << "Goal sample generated" << std::endl;
            } else {
                sample = env_bbox.sample();
            }
            //std::cout << fmt::format("Sample: [{:.2f}, {:.2f}, {:.2f}]", sample(0), sample(1), sample(2)) << std::endl;

            // If the sample is invalid, go to the next iteration
            if (!isStateValid(sample)){
                //std::cout << "Sample is not valid" << std::endl;
                if (sample == cur_goal_location_){
                    throw std::invalid_argument("Goal location has to be valid!");
                }
                continue;
            }
            //std::cout << "Sample is valid" << std::endl;

            // Get nearest node
            V steer_vertex;
            auto nearest_vertex = nn.nearest(sample);
            if (nearest_vertex.has_value()){
                // std::cout << "Checking collision" << std::endl;
                checkLineSegmentCollision(nearest_vertex->first->value_, sample, steer_vertex);
                if (steer_vertex == cur_goal_location_){
                    //std::cout << "Solution found!" << std::endl;
                    // Solution is found
                    num_of_solutions_++;
                    auto prev_solution_ptr = solution_node_ptr_;
                    auto prev_solution_cost = cur_solution_cost_;
                    solution_node_ptr_ = nearest_vertex->first;
                    cur_solution_cost_ = getSolutionCost();
                    if (prev_solution_ptr != nullptr && cur_solution_cost_ > prev_solution_cost){
                        // Restore if the current solution is worse than previous one
                        solution_node_ptr_ = prev_solution_ptr;
                        cur_solution_cost_ = prev_solution_cost;
                        return;
                    }
                } else if (steer_vertex != nearest_vertex->first->value_) {
                    //std::cout << "Solution is not found!" << std::endl;
                    //std::cout << "Printing steer vertex" << std::endl;
                    //std::cout << steer_vertex << std::endl;
                    auto steer_vertex_ptr = nearest_vertex->first->insertChild(steer_vertex);
                    nn.insert(steer_vertex_ptr);
                    //if (checkLineSegmentCollision(nearest_vertex->first->value_, steer_vertex_ptr->value_)){
                    //    throw std::invalid_argument("Line segment between two vertices collide in the planning phase!");
                    //}
                    num_of_tree_nodes_++;
                }
                iteration_idx++;
            }
        }
    }
}