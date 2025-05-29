//
// Created by peter on 3/10/22.
//
// Note: If part of the global point cloud is not visible in the visualizer
// 1) Increase block count
// 2) Decrease truncation weight

#include <atomic>
#include <sstream>
#include <thread>
#include "open3d/Open3D.h"
#include "dataset_utils/dataset_utils.h"
//#include "../libs/open3d_pcl_conversion/include/convert.h"
#include "gmm_map/Visualization.h"
#include "gmm_map/map_param.h"
#include "gmm_map/map_param_init.h"

using namespace open3d;
using namespace open3d::visualization;
namespace fs = std::filesystem;

// Tanglo colorscheme (see https://en.wikipedia.org/wiki/Tango_Desktop_Project)
static const Eigen::Vector3d kTangoOrange(0.961, 0.475, 0.000); // Color for frustum
static const Eigen::Vector3d kTangoSkyBlueDark(0.125, 0.290, 0.529); // Color for trajectory
static const Eigen::Vector3d kTangoRed(0.921, 0.203, 0.203); // Color for icp trajectory
static const Eigen::Vector3d kTangoGreen(0.074, 0.831, 0.353); // Color for ndt reg trajectory
//static const Eigen::Vector3d kOccOne(0.01, 0.01, 0.01); // Color for gmm clusters & trajectory
//static const Eigen::Vector3d kOccZero(0.5, 0.5, 0.9); // Color for gmm clusters & trajectory

// Colormap selection
static const std::string OccVarColormapClass = "diverging";
static const std::string OccVarColormapName = "BuYlRd";

//------------------------------------------------------------------------------
// Configure sliders and property panel
class PropertyPanel : public gui::VGrid {
    // Set the locations of the property panels
    using Super = gui::VGrid;

public:
    // num_columns: specify the width of the property panel
    // spacing: specify the spacing between successive columns
    // margin: set margins for gui within the columns
    PropertyPanel(int spacing, int left_margin)
            : gui::VGrid(2, spacing, gui::Margins(left_margin, 0, 0, 0)) {
        default_label_color_ =
                std::make_shared<gui::Label>("temp")->GetTextColor();
    }

    // Configure a toggle (true/false) slider
    // The value of this slider toggles the boolean value in the bool_addr
    void AddBool(const std::string& name,
                 std::atomic<bool>* bool_addr,
                 bool default_val,
                 const std::string& tooltip = "") {
        auto cb = std::make_shared<gui::Checkbox>("");
        cb->SetChecked(default_val);
        *bool_addr = default_val;
        cb->SetOnChecked([bool_addr, this](bool is_checked) {
            *bool_addr = is_checked;
            this->NotifyChanged();
        });
        auto label = std::make_shared<gui::Label>(name.c_str());
        label->SetTooltip(tooltip.c_str());
        AddChild(label);
        AddChild(cb);
    }

    // Configure a (double) floating point slider
    // The value of this slider toggles the boolean value in the num_addr
    void AddFloatSlider(const std::string& name,
                        std::atomic<gmm::FP>* num_addr,
                        gmm::FP default_val,
                        gmm::FP min_val,
                        gmm::FP max_val,
                        const std::string& tooltip = "") {
        auto s = std::make_shared<gui::Slider>(gui::Slider::DOUBLE);
        s->SetLimits(min_val, max_val);
        s->SetValue(default_val);
        *num_addr = default_val;
        s->SetOnValueChanged([num_addr, this](double new_val) {
            *num_addr = new_val;
            this->NotifyChanged();
        });
        auto label = std::make_shared<gui::Label>(name.c_str());
        label->SetTooltip(tooltip.c_str());
        AddChild(label);
        AddChild(s);
    }

    // Configure an int slider
    // The value of this slider toggles the boolean value in the num_addr
    void AddIntSlider(const std::string& name,
                      std::atomic<int>* num_addr,
                      int default_val,
                      int min_val,
                      int max_val,
                      const std::string& tooltip = "") {
        auto s = std::make_shared<gui::Slider>(gui::Slider::INT);
        s->SetLimits(min_val, max_val);
        s->SetValue(default_val);
        *num_addr = default_val;
        s->SetOnValueChanged([num_addr, this](int new_val) {
            *num_addr = new_val;
            this->NotifyChanged();
        });
        auto label = std::make_shared<gui::Label>(name.c_str());
        label->SetTooltip(tooltip.c_str());
        AddChild(label);
        AddChild(s);
    }

    void AddValues(const std::string& name,
                   std::atomic<int>* idx_addr,
                   int default_idx,
                   std::vector<std::string> values,
                   const std::string& tooltip = "") {
        auto combo = std::make_shared<gui::Combobox>();
        for (auto& v : values) {
            combo->AddItem(v.c_str());
        }
        combo->SetSelectedIndex(default_idx);
        *idx_addr = default_idx;
        combo->SetOnValueChanged(
                [idx_addr, this](const char* new_value, int new_idx) {
                    *idx_addr = new_idx;
                    this->NotifyChanged();
                });
        auto label = std::make_shared<gui::Label>(name.c_str());
        label->SetTooltip(tooltip.c_str());
        AddChild(label);
        AddChild(combo);
    }

    void SetEnabled(bool enable) override {
        Super::SetEnabled(enable);
        for (auto child : GetChildren()) {
            child->SetEnabled(enable);
            auto label = std::dynamic_pointer_cast<gui::Label>(child);
            if (label) {
                if (enable) {
                    label->SetTextColor(default_label_color_);
                } else {
                    label->SetTextColor(gui::Color(0.5f, 0.5f, 0.5f, 1.0f));
                }
            }
        }
    }

    // Callback function if slider values change
    void SetOnChanged(std::function<void()> f) { on_changed_ = f; }

private:
    gui::Color default_label_color_;
    std::function<void()> on_changed_;

    void NotifyChanged() {
        if (on_changed_) {
            on_changed_();
        }
    }
};

//------------------------------------------------------------------------------
class ReconstructionWindow : public gui::Window {
    using Super = gui::Window;

public:
    ReconstructionWindow(const std::string& dataset_name,
                         const int& start_frame_idx,
                         const int& num_frames,
                         const float& depth_pixel_variance,
                         const std::string& device,
                         gui::FontId monospace)
            : gui::Window("GMMap Construction (CPU)",
                          2 * dataset_param::vis_window_width,
                          2 * dataset_param::vis_window_height),
              dataset_name_(dataset_name),
              //intrinsic_path_(intrinsic_path),
              device_str_(device),
              is_running_(false),
              is_started_(false),
              is_terminated_(false),
              monospace_(monospace) {

        // Store visualization and GMM parameters
        this->start_frame_idx = start_frame_idx;
        this->num_frames = num_frames;
        this->depth_pixel_variance = depth_pixel_variance;

        ////////////////////////////////////////
        /// General layout
        auto& theme = GetTheme();
        int em = theme.font_size;
        int spacing = int(std::round(0.25f * float(em)));
        int left_margin = em;
        int vspacing = int(std::round(0.5f * float(em)));
        gui::Margins margins(int(std::round(0.5f * float(em))));
        panel_ = std::make_shared<gui::Vert>(spacing, margins);
        widget3d_ = std::make_shared<gui::SceneWidget>();
        fps_panel_ = std::make_shared<gui::Vert>(spacing, margins);

        // Add left panel (panel_), 3D visualization space (widget3d_), and fps panel (fps_panel_)
        AddChild(panel_);
        AddChild(widget3d_);
        AddChild(fps_panel_);

        ////////////////////////////////////////
        /// Property panels
        // Create sliders for the property panel (panel_)
        // Note that the values are stored in struct (prop_values_)
        // Add fixed setting before reconstruction
        fixed_props_ = std::make_shared<PropertyPanel>(spacing, left_margin);
        // Set default scaling parameters
        gmm::FP default_depth_scale = 1;
        double default_voxel_size = 0.03;
        if (dataset_name_ == "tum" || dataset_name_ == "tum_fd" || dataset_name_ == "stata"){
            default_depth_scale = dataset_param::scale;
            default_voxel_size = 0.01;
        } else if (dataset_name_ == "tartanair"){
            default_depth_scale = 1000;
        }
        prop_values_.depth_max = dataset_param::max_depth;
        fixed_props_->AddFloatSlider("Measurement Scale", &prop_values_.depth_scale,
                                   default_depth_scale, 1, 8000,
                                   "Scale factor applied to the depth values "
                                   "from the depth image (except .npy files).");
        fixed_props_->AddFloatSlider("Point Voxel Size (m)", &prop_values_.voxel_size,
                                     default_voxel_size, 0.01, 0.5,
                                     "Voxel size for the TSDF voxel grid.");
        fixed_props_->AddFloatSlider(
                "Trunc multiplier", &prop_values_.trunc_voxel_multiplier, 1.0,
                1.0, 20.0,
                "Truncate distance multiplier (in voxel size) to control "
                "the volumetric surface thickness.");
        fixed_props_->AddIntSlider(
                "Block Count", &prop_values_.bucket_count, 70000, 10000, 100000,
                "Number of estimated voxel blocks for spatial "
                "hashmap. Will be adapted dynamically, but "
                "may trigger memory issue during rehashing for large scenes.");
        fixed_props_->AddIntSlider(
                "Estimated Points", &prop_values_.pointcloud_size, 30720000,
                10000000, 80000000,
                "Estimated number of points in the point cloud; used to speed "
                "extraction of points into the 3D scene.");
        fixed_props_->AddBool("Map Obstacles",
                              &prop_values_.update_obs_gmms, true,
                              "Map obstacles in the environment");
        fixed_props_->AddBool("Map Free Space",
                              &prop_values_.update_free_gmms, true,
                              "Map free space in the environment");
        fixed_props_->AddBool("Fuse GMM across frames",
                              &prop_values_.fuse_gmm_across_frames, true,
                              "Local or global GMM");

        // Add reconstruction settings
        adjustable_props_ =
                std::make_shared<PropertyPanel>(spacing, left_margin);
        adjustable_props_->AddFloatSlider("Point Cloud Truncation Weight", &prop_values_.truncation_weight,
                                     1.0f, 0.0, 5.0f,
                                     "Truncation weight associated with the point cloud model.");
        adjustable_props_->AddIntSlider(
                "Update Interval", &prop_values_.surface_interval, dataset_param::vis_updated_interval, 1, 500,
                "The number of iterations between updating the 3D display.");
        adjustable_props_->AddBool("Update Surface",
                                   &prop_values_.update_surface, true,
                                   "Update surface every several frames, "
                                   "determined by the update interval.");
        adjustable_props_->AddIntSlider(
                "Visualization Latency", &prop_values_.sleep_ms, dataset_param::vis_delay_per_frame, 0, 2000,
                "Frame to frame latency for constructing the map (ms).");
        adjustable_props_->AddBool(
                "Enable Camera Tracking", &prop_values_.enable_camera_tracking, true,
                "Tracking the robot trajectory");
        adjustable_props_->AddFloatSlider("Current Viewpoint Weight", &prop_values_.cur_viewpoint_weight,
                                          0.1, 0.01, 0.5,
                                          "Weight assigned to current frame for smoothing tracking");
        adjustable_props_->AddFloatSlider("Camera Zoom Factor", &prop_values_.zoom_factor,
                                          dataset_param::vis_zoom_factor, 0.1, 3.0,
                                          "Camera zoom factor for trajectory tracking.");
        adjustable_props_->AddFloatSlider("Camera Height Factor", &prop_values_.height_factor,
                                          dataset_param::vis_height_factor, 0.1, 2.0,
                                          "Camera pivot height factor for trajectory tracking.");
        adjustable_props_->AddBool("Save Map",  &prop_values_.save_map, false, "Saved GMMap into a binary file");
        adjustable_props_->AddBool("Save Occ Images",  &prop_values_.save_occ_img, false, "Saved occupancy distribution at a cross section");

        // Add geometry settings
        geometry_props_ = std::make_shared<PropertyPanel>(spacing, left_margin);
        geometry_props_->AddBool("Show Pointcloud",
                                   &prop_values_.show_pcd, true,
                                   "Pointcloud visualization");
        geometry_props_->AddBool(
                "Show Gndtruth Trajectory", &prop_values_.show_traj, true,
                "Trajectory visualization");
        geometry_props_->AddBool("Show Axes",
                                   &prop_values_.show_coord, false,
                                   "Axes visualization at origin");
        geometry_props_->AddBool(
                "Show GMM Obstacles", &prop_values_.atomicMapVizFlags.show_gmm_obs, dataset_param::vis_show_obstacle_gmm,
                "GMM obstacles visualization");
        geometry_props_->AddBool(
                "Show GMM Free", &prop_values_.atomicMapVizFlags.show_gmm_free, dataset_param::vis_show_free_gmm,
                "GMM free visualization");
        geometry_props_->AddBool(
                "Show GMM Free (Near Obs)", &prop_values_.atomicMapVizFlags.show_gmm_free_near_obs, dataset_param::vis_show_free_gmm,
                "GMM free visualization");
        geometry_props_->AddBool(
                "Show GMM Color", &prop_values_.atomicMapVizFlags.show_gmm_color, false,
                "GMM colored visualization");
        geometry_props_->AddBool(
                "Show Obstacle BBox", &prop_values_.atomicMapVizFlags.show_gmm_bbox_obs, false,
                "GMM obstable bounding box visualization");
        geometry_props_->AddBool(
                "Show Free BBox", &prop_values_.atomicMapVizFlags.show_gmm_bbox_free, false,
                "GMM free bounding box visualization");
        geometry_props_->AddBool(
                "Show Global BBox", &prop_values_.show_env_bbox, false,
                "Bounding-box visualization");
        geometry_props_->AddBool(
                "Show Occ Pts", &prop_values_.atomicMapVizFlags.show_occupancy_pts, false,
                "Occupancy Samples");
        geometry_props_->AddBool(
                "Show Var Pts", &prop_values_.atomicMapVizFlags.show_variance_pts, false,
                "Variance Samples");
        geometry_props_->AddBool(
                "Show Voxels", &prop_values_.atomicMapVizFlags.show_occupancy_voxels, false,
                "Occupancy Voxels");
        geometry_props_->AddFloatSlider("Voxel Resolution", &prop_values_.atomicMapVizFlags.voxel_resolution,
                                        0.1, 0.01, 0.5,
                                        "Occupancy Voxel Resolution");
        geometry_props_->AddFloatSlider("Lower Bound", &prop_values_.atomicMapVizFlags.occ_var_pt_low,
                                        0.0, 0.0, 1.0,
                                        "Lower bound");
        geometry_props_->AddFloatSlider("Upper Bound", &prop_values_.atomicMapVizFlags.occ_var_pt_high,
                                        1.0, 0.0, 1.0,
                                        "Upper bound");
        geometry_props_->AddFloatSlider("Gaussian std", &prop_values_.atomicMapVizFlags.std,
                                        2.0, 0.1, 4.0,
                                        "Standard deviation for displaying Gaussian ellipsoids");
        geometry_props_->AddIntSlider("Number of Samples", &prop_values_.atomicMapVizFlags.num_of_points,
                                        1000000, 1000, 10000000,
                                        "Number of samples of evaluation occupancy and variance within the bounding box");
        prop_values_.atomicMapVizFlags.from_ray = false;

        // Add geometry settings
        en_vis_props_ = std::make_shared<PropertyPanel>(spacing, left_margin);
        en_vis_props_->AddBool("Show Env BBox", &prop_values_.atomicMapVizFlags.show_env_bbox, true,
                "Bounding box of the occupied and free GMMs");
        en_vis_props_->AddFloatSlider("x-axis Lower Bound", &prop_values_.atomicMapVizFlags.env_bbox_low_x,
                                        0.0, 0.0, 1.0,
                                        "x-axis lower bound");
        en_vis_props_->AddFloatSlider("x-axis Upper Bound", &prop_values_.atomicMapVizFlags.env_bbox_high_x,
                                        1.0, 0.0, 1.0,
                                        "x-axis Upper Bound");
        en_vis_props_->AddFloatSlider("y-axis Lower Bound", &prop_values_.atomicMapVizFlags.env_bbox_low_y,
                                      0.0, 0.0, 1.0,
                                      "y-axis lower bound");
        en_vis_props_->AddFloatSlider("y-axis Upper Bound", &prop_values_.atomicMapVizFlags.env_bbox_high_y,
                                      1.0, 0.0, 1.0,
                                      "y-axis Upper Bound");
        en_vis_props_->AddFloatSlider("z-axis Lower Bound", &prop_values_.atomicMapVizFlags.env_bbox_low_z,
                                      0.0, 0.0, 1.0,
                                      "z-axis lower bound");
        en_vis_props_->AddFloatSlider("z-axis Upper Bound", &prop_values_.atomicMapVizFlags.env_bbox_high_z,
                                      1.0, 0.0, 1.0,
                                      "z-axis Upper Bound");

        // Add to panels
        panel_->AddChild(std::make_shared<gui::Label>("Starting Settings"));
        panel_->AddChild(fixed_props_);
        panel_->AddFixed(vspacing);
        panel_->AddChild(std::make_shared<gui::Label>("Reconstruction Settings"));
        panel_->AddChild(adjustable_props_);
        panel_->AddFixed(vspacing);
        panel_->AddChild(std::make_shared<gui::Label>("Visualization Settings"));
        panel_->AddChild(geometry_props_);
        panel_->AddChild(std::make_shared<gui::Label>("Environment Boundary Settings"));
        panel_->AddChild(en_vis_props_);


        // Create pause and resume toggle and set callback functions
        auto b = std::make_shared<gui::ToggleSwitch>("Pause/Resume");
        b->SetOnClicked([b, this](bool is_on) {
            // Defining a callback function (used for initialization to start the process)
            if (!this->is_started_) {
                gui::Application::GetInstance().PostToMainThread(
                        this, [this]() {
                            int max_points = prop_values_.pointcloud_size;
                            t::geometry::PointCloud pcd_placeholder(
                                    core::Tensor({max_points, 3},
                                                 core::Dtype::Float32,
                                                 core::Device("CPU:0")));
                            pcd_placeholder.SetPointColors(core::Tensor(
                                    {max_points, 3}, core::Dtype::Float32,
                                    core::Device("CPU:0")));

                            auto mat = rendering::MaterialRecord();
                            mat.shader = "defaultUnlit";
                            //mat.sRGB_vertex_color = true;
                            this->widget3d_->GetScene()
                                    ->GetScene()
                                    ->AddGeometry("points", pcd_placeholder,
                                                  mat);

                            this->trajectory_ = std::make_shared<camera::PinholeCameraTrajectory>();
                            this->icp_trajectory_ = std::make_shared<camera::PinholeCameraTrajectory>();
                            this->ndt_trajectory_ = std::make_shared<camera::PinholeCameraTrajectory>();
                            this->gmm_trajectory_ = std::make_shared<camera::PinholeCameraTrajectory>();

                            float voxel_size = prop_values_.voxel_size;
                            this->model_ = std::make_shared<t::pipelines::slam::Model>(
                                    voxel_size, 16,
                                    prop_values_.bucket_count,
                                    core::Tensor::Eye(
                                            4, core::Dtype::Float64,
                                            core::Device("CPU:0")),
                                    core::Device(device_str_));
                            this->is_started_ = true;
                        });
            }
            this->is_running_ = !(this->is_running_);
            this->adjustable_props_->SetEnabled(true);
        });
        panel_->AddChild(b);
        panel_->AddFixed(vspacing);
        auto b_save = std::make_shared<gui::ToggleSwitch>("Toggle to Capture Screen");
        b_save->SetOnClicked([b_save, this](bool is_on) {
            // Defining a callback function (used for initialization to start the process)
            if (this->is_started_) {
                //std::atomic<bool> done_redraw_flag = false;
                gui::Application::GetInstance().PostToMainThread(
                        this, [this, result_path = dataset_param::result_path
                                //flag_ptr = &done_redraw_flag
                                        ]() {
                            time_t rawtime;
                            struct tm * timeinfo;
                            char buffer[80];
                            time (&rawtime);
                            timeinfo = localtime(&rawtime);
                            strftime(buffer,80,"%d-%m-%Y-%H-%M-%S",timeinfo);
                            std::string cur_time  = std::string(buffer);
                            //std::cout << fmt::format("Saved image to file: {}", cur_time) << std::endl;
                            std::function<void(std::shared_ptr< geometry::Image>)> img_cb = [cur_time, result_path] (const std::shared_ptr< geometry::Image>& img)
                            {
                                std::string filename = result_path / fmt::format("{}.jpg", cur_time);
                                open3d::io::WriteImageToJPG(filename, *img);
                                std::cout << fmt::format("Saved current screen to file: {}", filename) << std::endl;
                            };
                            //this->widget3d_->ForceRedraw();
                            this->widget3d_->GetScene()->GetScene()->RenderToImage(img_cb);
                            //*flag_ptr = true;
                        });
                //while(!done_redraw_flag){
                //    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                //}
            }
        });

        panel_->AddChild(b_save);
        panel_->AddFixed(vspacing);

        auto b_saveViewpoint = std::make_shared<gui::ToggleSwitch>("Toggle To Save Viewpoint");
        b_saveViewpoint->SetOnClicked([b, this](bool is_on) {
            // Defining a callback function (used for initialization to start the process)
            gui::Application::GetInstance().PostToMainThread(
                    this, [this, result_path = dataset_param::result_path]() {
                        auto model_matrix = this->widget3d_->GetRenderView()->GetCamera()->GetModelMatrix();
                        std::string viewpoint_file = result_path / "viewpoint_matrix.txt";
                        dutil::writeEigenToCSV(viewpoint_file, model_matrix.matrix().cast<double>());
                        std::cout << fmt::format("Saved viewpoint to file: {}", viewpoint_file) << std::endl;
                    });
        });
        panel_->AddChild(b_saveViewpoint);
        panel_->AddFixed(vspacing);

        auto b_loadViewpoint = std::make_shared<gui::ToggleSwitch>("Toggle To Load Viewpoint");
        b_loadViewpoint->SetOnClicked([b, this](bool is_on) {
            // Defining a callback function (used for initialization to start the process)
            gui::Application::GetInstance().PostToMainThread(
                    this, [this, result_path = dataset_param::result_path]() {
                        std::string viewpoint_file = result_path / "viewpoint_matrix.txt";
                        if (std::filesystem::exists(viewpoint_file)){
                            Eigen::Isometry3f model_matrix_from_file;
                            model_matrix_from_file.matrix() = dutil::readEigenFromCSV(viewpoint_file).cast<float>();
                            this->widget3d_->GetRenderView()->GetCamera()->SetModelMatrix(model_matrix_from_file);
                            std::cout << fmt::format("Viewpoint loaded from file: {}", viewpoint_file) << std::endl;
                        } else {
                            std::cout << fmt::format("Viewpoint file {} do not exist!", viewpoint_file) << std::endl;
                        }
                    });
        });
        panel_->AddChild(b_loadViewpoint);
        panel_->AddFixed(vspacing);
        panel_->AddStretch();

        ////////////////////////////////////////
        /// Tabs

        gui::Margins tab_margins(0, int(std::round(0.5f * float(em))), 0, 0);
        auto tabs = std::make_shared<gui::TabControl>();
        panel_->AddChild(tabs);

        // Create a tab that displays images from RGB and depth images
        auto tab1 = std::make_shared<gui::Vert>(0, tab_margins);
        input_color_image_ = std::make_shared<gui::ImageWidget>();
        input_depth_image_ = std::make_shared<gui::ImageWidget>();
        tab1->AddChild(input_color_image_);
        tab1->AddFixed(vspacing);
        tab1->AddChild(input_depth_image_);
        tabs->AddTab("Input images", tab1);

        // Create an info tab that displays the pose and other informations about the reconstruction system
        auto tab3 = std::make_shared<gui::Vert>(0, tab_margins);
        output_info_ = std::make_shared<gui::Label>("");
        output_info_->SetFontId(monospace_);
        tab3->AddChild(output_info_);
        tabs->AddTab("Info", tab3);

        // Initialize 3D visualization tab
        widget3d_->SetScene(std::make_shared<rendering::Open3DScene>(GetRenderer()));

        // Configure FPS panel (default)
        output_fps_ = std::make_shared<gui::Label>("Calculating Bounding Box ...\nCheck terminal!");
        fps_panel_->AddChild(output_fps_);

        is_done_ = false;
        // Describe behaviour after the reconstruction is completed / window is closed
        SetOnClose([this]() {
            is_done_ = true;
            is_terminated_ = true;
            if (is_started_) {
                utility::LogInfo("Writing trajectory to trajectory.log...");
                io::WritePinholeCameraTrajectory("gndtruth_trajectory.log",*trajectory_);
                utility::LogInfo("Done!");
            }
            std::cout << "Closing window and terminate!" << std::endl;
            gui::Application::GetInstance().RemoveWindow(this);
            gui::Application::GetInstance().Quit();
            return true;  // false would cancel the close
        });
        // Construct a new thread for updating the visualizer
        update_thread_ = std::thread([this]() { this->UpdateMain(); });
    }

    // Destructor
    ~ReconstructionWindow() { update_thread_.join(); }

    // Describe the location of the three panels that were defined.
    void Layout(const gui::LayoutContext& context) override {
        int em = context.theme.font_size;
        int panel_width = 20 * em;
        // The usable part of the window may not be the full size if there
        // is a menu.
        auto content_rect = GetContentRect();
        panel_->SetFrame(gui::Rect(content_rect.x, content_rect.y, panel_width,
                                   content_rect.height));
        int x = panel_->GetFrame().GetRight();
        widget3d_->SetFrame(gui::Rect(x, content_rect.y,
                                      content_rect.GetRight() - x,
                                      content_rect.height));

        int fps_panel_width = 14 * em;
        int fps_panel_height = 7 * em;
        fps_panel_->SetFrame(
                gui::Rect(content_rect.GetRight() - fps_panel_width,
                          content_rect.y, fps_panel_width, fps_panel_height));

        // Now that all the children are sized correctly, we can super to
        // layout all their children.
        Super::Layout(context);
    }

    // Set the information displayed in the info tab
    void SetInfo(const std::string& output) {
        output_info_->SetText(output.c_str());
    }

    // Set the FPS in the panel
    void SetFPS(const std::string& output) {
        output_fps_->SetText(output.c_str());
    }

    void updateInfoAndFPS(std::stringstream& info, std::stringstream& fps,
                          const int& cur_frame_idx, const int& max_num_frames, const Matrix4d& T_eigen, gmm::GMMMapViz* gmm_map_viz,
                          const double& visualizer_tp){
        info.str(std::string());
        fps.str(std::string());
        gmm::FP cluster_size, rtree_size;
        int num_rtree_nodes;
        gmm_map_viz->gmm_map->estimateMapSize(cluster_size, rtree_size, num_rtree_nodes);

        info << fmt::format("Frame {}/{}\n\n", cur_frame_idx + 1, max_num_frames);

        info << "GndTruth Transformation:\n";
        // Formating
        Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[","]");
        info << T_eigen.format(CleanFmt) << "\n\n";

        info << fmt::format("Active voxel blocks: {}/{}\n",
                            model_->GetHashMap().Size(),
                            model_->GetHashMap().GetCapacity());
        {
            std::lock_guard<std::mutex> locker(surface_.lock);
            int64_t len = surface_.pcd_cropped.HasPointPositions()
                          ? surface_.pcd_cropped.GetPointPositions().GetLength()
                          : 0;
            info << fmt::format("Surface points: {}/{}\n", len,
                                prop_values_.pointcloud_size)
                 << "\n";
        }

        std::string fps_str;
        {
            const std::lock_guard<std::mutex> g_lock(*gmm_map_viz->geometry_lock_ptr);
            int total_sampling_pts = gmm_map_viz->total_num_query_pts;
            info << fmt::format("Obstacle clusters: {}\nFree clusters: {}\n",
                                gmm_map_viz->cur_map_updates.num_obs_clusters, gmm_map_viz->cur_map_updates.num_free_clusters);
            info << fmt::format("Visualized Gaussians: {}/{}\n",
                                gmm_map_viz->gmm_linesets.active_free_gaussians.size() +
                                gmm_map_viz->gmm_linesets.active_obs_gaussians.size() +
                                gmm_map_viz->gmm_linesets.active_free_near_obs_gaussians.size(),
                                gmm_map_viz->gmm_linesets.gaussians.size());
            info << fmt::format("Sampling accuracy: {}/{} ({:.2f})\nOccupancy variance: [{:.2f}, {:.2f}]\n",
                                (int) (gmm_map_viz->query_accuracy*total_sampling_pts), total_sampling_pts, gmm_map_viz->query_accuracy,
                                gmm_map_viz->cur_min_variance, gmm_map_viz->cur_max_variance);
            info << fmt::format("Total cluster size: {:.2f}KB\nRtree size: {:.2f}KB,\nTotal Rtree nodes: {}\n",
                                cluster_size, rtree_size, num_rtree_nodes);
            fps_str = fmt::format("Visualization FPS: {:.3f}\nGMM Clustering FPS: {:.3f}\nGMM Fusion FPS: {:.3f}\nGMM Mapping FPS: {:.3f}\nSampling Tp: {:.3e}pts/s\nMap Size: {:.2f}KB",
                                  visualizer_tp, gmm_map_viz->cur_map_updates.gmm_clustering_tp, gmm_map_viz->cur_map_updates.gmm_fusion_tp,
                                  gmm_map_viz->cur_map_updates.gmm_mapping_tp, gmm_map_viz->query_throughput,
                                  cluster_size + rtree_size);
        }

        info << fps_str;
        fps << fps_str;
    }

protected:
    std::string dataset_name_;
    std::string device_str_;

    // General GMM parameters and visualization options
    int start_frame_idx;
    int num_frames;
    float depth_pixel_variance;

    // General logic
    std::atomic<bool> is_running_;
    std::atomic<bool> is_started_;
    std::atomic<bool> is_done_;
    std::atomic<bool> is_terminated_;

    // Panels and controls
    gui::FontId monospace_;
    std::shared_ptr<gui::Vert> panel_;
    std::shared_ptr<gui::Label> output_info_;
    std::shared_ptr<PropertyPanel> fixed_props_;
    std::shared_ptr<PropertyPanel> adjustable_props_;
    std::shared_ptr<PropertyPanel> geometry_props_;
    std::shared_ptr<PropertyPanel> en_vis_props_;

    std::shared_ptr<gui::SceneWidget> widget3d_;

    std::shared_ptr<gui::Vert> fps_panel_;
    std::shared_ptr<gui::Label> output_fps_;

    // Image Widgets
    std::shared_ptr<gui::ImageWidget> input_color_image_;
    std::shared_ptr<gui::ImageWidget> input_depth_image_;
    std::shared_ptr<gui::ImageWidget> raycast_color_image_;
    std::shared_ptr<gui::ImageWidget> raycast_depth_image_;

    // Slider parameters set before reconstruction
    struct {
        std::atomic<int> surface_interval;
        std::atomic<int> pointcloud_size;
        std::atomic<gmm::FP> depth_scale;
        std::atomic<int> bucket_count;
        std::atomic<gmm::FP> voxel_size;
        std::atomic<gmm::FP> truncation_weight;
        std::atomic<gmm::FP> trunc_voxel_multiplier;
        std::atomic<gmm::FP> depth_max;
        std::atomic<gmm::FP> depth_diff;
        std::atomic<bool> raycast_color;
        std::atomic<bool> raycast_depth;
        std::atomic<bool> update_surface;
        std::atomic<bool> save_map;
        std::atomic<bool> save_occ_img;

        std::atomic<bool> update_obs_gmms;
        std::atomic<bool> update_free_gmms;
        std::atomic<bool> fuse_gmm_across_frames;

        std::atomic<int> sleep_ms;

        gmm::vizMapAtomicFlags atomicMapVizFlags;

        // User added fields
        std::atomic<bool> show_pcd;
        std::atomic<bool> show_traj; // Ground truth
        std::atomic<bool> show_icp_traj; // ICP
        std::atomic<bool> show_gmm_traj; // GMM
        std::atomic<bool> show_ndt_traj; // NDT
        std::atomic<bool> show_coord;
        std::atomic<bool> show_env_bbox;

        std::atomic<bool> enable_camera_tracking;
        std::atomic<gmm::FP> cur_viewpoint_weight;
        std::atomic<gmm::FP> zoom_factor;
        std::atomic<gmm::FP> height_factor;

        // Evaluation setting
        std::atomic<gmm::FP> ray_sampling_dist;
    } prop_values_;

    struct {
        std::mutex lock;
        geometry::PointCloud pcd;
        t::geometry::PointCloud pcd_cropped;
    } surface_;
    std::atomic<bool> is_scene_updated_;
    std::mutex pcd_lock;

    // Geometries
    std::shared_ptr<t::pipelines::slam::Model> model_;
    std::shared_ptr<camera::PinholeCameraTrajectory> trajectory_; // Ground truth trajectory
    std::shared_ptr<camera::PinholeCameraTrajectory> icp_trajectory_; // Trajectory estimated from ICP registration
    std::shared_ptr<camera::PinholeCameraTrajectory> gmm_trajectory_; // Trajectory estimated from GMM registration
    std::shared_ptr<camera::PinholeCameraTrajectory> ndt_trajectory_; // Trajectory estimated from GMM registration
    // Thread control
    std::thread update_thread_;

protected:
    // Note that we cannot update the GUI on this thread, we must post to
    // the main thread! This function runs on a seperate thread
    void UpdateMain() {
        // Load files that constains a list for depth filenames and rgb filenames
        std::vector<std::string> rgb_files, depth_files, pose_files;
        std::vector<double> timestamps;
        dutil::LoadFilenames(rgb_files, depth_files, pose_files, timestamps, dataset_name_);

        /*
        for (int j = 0; j < rgb_files.size(); j++){
            std::cout << rgb_files.at(j) << std::endl;
            std::cout << depth_files.at(j) << std::endl;
            std::cout << pose_files.at(j) << std::endl;
        }
        */

        // Load intrinsics (if provided)
        core::Tensor intrinsic_t;
        camera::PinholeCameraIntrinsic intrinsic_legacy;
        // Obtain camera intrinsic parameters
        std::tie(intrinsic_t, intrinsic_legacy) = dutil::LoadIntrinsics(dataset_name_);
        camera::PinholeCameraParameters traj_param;
        traj_param.intrinsic_ = intrinsic_legacy;
        auto K_eigen = core::eigen_converter::TensorToEigenMatrixXd(intrinsic_t);

        // Only set at initialization (from the sliders in panels)
        // Initialize pose T tensor to the ground truth one
        core::Tensor T_frame_to_model, T_frame_to_model_icp, T_frame_to_model_gmm, T_frame_to_model_ndt;

        // Load initial results from ground truth pose
        dutil::processPoseO3d(pose_files, start_frame_idx, dataset_name_,T_frame_to_model);
        T_frame_to_model_icp = T_frame_to_model;
        T_frame_to_model_gmm = T_frame_to_model;
        T_frame_to_model_ndt = T_frame_to_model;

        core::Device device(device_str_);

        // Initialize depth and color channels
        std::cout << "Initializing depth and color channels ..." << std::endl;
        std::shared_ptr<t::geometry::Image> ref_depth, ref_color;
        if (dataset_name_ == "tartanair"){
            ref_depth = dutil::createImageNpy2O3d(depth_files[start_frame_idx], prop_values_.depth_scale);
        } else {
            ref_depth = t::io::CreateImageFromFile(depth_files[start_frame_idx]);
        }

        ref_color = t::io::CreateImageFromFile(rgb_files[start_frame_idx]);

        if (dataset_name_ == "stata"){
            *ref_depth = ref_depth->Resize(dataset_param::sampling_rate);
            *ref_color = ref_color->Resize(dataset_param::sampling_rate);
        }

        // Initialize voxel hased input frames (from actual depth map) and raycasted frames (from raycasting)
        std::cout << "Initializing voxel hashes ..." << std::endl;
        t::pipelines::slam::Frame input_frame(
                ref_depth->GetRows(), ref_depth->GetCols(), intrinsic_t, device);

        // Odometry & images needed for display!
        std::cout << "Initializing geometric objects ..." << std::endl;
        auto traj = std::make_shared<geometry::LineSet>();
        auto frustum = std::make_shared<geometry::LineSet>();

        auto icp_traj = std::make_shared<geometry::LineSet>();
        auto icp_frustum = std::make_shared<geometry::LineSet>();

        auto gmm_traj = std::make_shared<geometry::LineSet>();
        auto gmm_frustum = std::make_shared<geometry::LineSet>();

        auto ndt_traj = std::make_shared<geometry::LineSet>();
        auto ndt_frustum = std::make_shared<geometry::LineSet>();

        auto color = std::make_shared<geometry::Image>();
        auto depth_colored = std::make_shared<geometry::Image>();
        auto raycast_color = std::make_shared<geometry::Image>();
        auto raycast_depth_colored = std::make_shared<geometry::Image>();
        auto coord_axes = geometry::TriangleMesh::CreateCoordinateFrame(0.6);
        auto env_bbox = std::make_shared<geometry::AxisAlignedBoundingBox>();
        is_scene_updated_ = false;

        color = std::make_shared<geometry::Image>(
                ref_color->ToLegacy());
        depth_colored = std::make_shared<geometry::Image>(
                ref_depth->ColorizeDepth(prop_values_.depth_scale, 0.001, prop_values_.depth_max).ToLegacy());

        // Determine the size of environment
        //std::cout << "Computing bounding box of the environment ..." << std::endl;
        Eigen::Vector3d lowerBound, upperBound, map_size, map_center;
        if (!dutil::readBBoxFromCSV(dataset_name_, upperBound,lowerBound)){
            dutil::obtainEnvBBox(dataset_name_, prop_values_.depth_max, prop_values_.depth_scale, lowerBound, upperBound);
            dutil::saveBBoxToCSV(dataset_name_, upperBound,lowerBound);
        }

        map_size = upperBound - lowerBound;
        map_center = lowerBound + map_size/2;
        env_bbox = std::make_shared<geometry::AxisAlignedBoundingBox>(geometry::AxisAlignedBoundingBox(lowerBound, upperBound));
        env_bbox->color_ << 0, 0, 0;
        float invert_z = 1;
        if (dataset_name_ == "stata" || dataset_name_ == "tartanair"){
            invert_z = -1;
        }

        // Render once to refresh (Initialization only!)
        std::cout << "Displaying initialized geometric objects ..." << std::endl;
        // Update display by referring to the main thread
        gui::Application::GetInstance().PostToMainThread(
                this, [this, color, depth_colored, raycast_color,
                        raycast_depth_colored, coord_axes, env_bbox, invert_z]() {
                    // Set displays for all panels / tabs
                    this->input_color_image_->UpdateImage(color);
                    this->input_depth_image_->UpdateImage(depth_colored);
                    this->SetNeedsLayout();  // size of image changed

                    // Add coordinate axes
                    auto mat = rendering::MaterialRecord();
                    mat.shader = "defaultUnlit";
                    //mat.line_width = 5.0f;
                    this->widget3d_->GetScene()->AddGeometry(
                            "coord_axes", coord_axes.get(), mat);

                    // Set views for the reconstruction
                    // 1) Define the BBox that bounds the entire scene
                    //geometry::AxisAlignedBoundingBox bbox(
                    //        Eigen::Vector3d(-27, -4, -16),
                    //        Eigen::Vector3d(0, 4, 11));
                    // 2) Obtain the center of such box
                    auto center = env_bbox->GetCenter().cast<float>();
                    // 3) Setup the camera parameters (fov, scene size, center of the rotation)
                    this->widget3d_->SetupCamera(60, *env_bbox, center);
                    // Setup view angle of the camera (view location, camera location, up direction)
                    this->widget3d_->LookAt(center,
                                            Eigen::Vector3f{10, 10, invert_z*10},
                                            {0.0f, 0.0f, invert_z});

                    // Display BBox
                    mat.shader = "unlitLine";
                    mat.line_width = 5.0f;
                    // https://github.com/isl-org/Open3D/issues/2890
                    // Only need to change absorption distance and color!
                    /*
                    mat.shader = "defaultLitSSR";
                    mat.base_color << 0, 0, 0, 1;
                    mat.base_roughness = 0.0;
                    mat.base_reflectance = 1.0;
                    mat.base_clearcoat = 0.0;
                    mat.thickness = 5;
                    mat.transmission = 1.0;
                    mat.absorption_distance = 5;
                    mat.absorption_color << 0, 1, 0;
                    auto test_box = open3d::geometry::TriangleMesh::CreateBox(5,5,5);
                     */
                    this->widget3d_->GetRenderView()->SetPostProcessing(false);
                    this->widget3d_->GetScene()->AddGeometry("bbox", env_bbox.get(), mat);
                    this->widget3d_->GetScene()->GetScene()->EnableIndirectLight(false);
                    this->widget3d_->GetScene()->GetScene()->ShowSkybox(false);
                    Eigen::Vector4f background_color;
                    //background_color << 0.7, 0.7, 0.7, 1;
                    background_color << 1, 1, 1, 1;
                    this->widget3d_->GetScene()->SetBackground(background_color);
                });

        Eigen::IOFormat CleanFmt(Eigen::StreamPrecision, 0, ", ", "\n", "[",
                                 "]");

        const int fps_interval_len = 10; // Number of frames to update FPS
        int total_fps_interval_len = 0;
        double time_interval = 0;
        double time_icp_interval = 0;
        double time_ndt_interval = 0;
        size_t idx = 0;


        auto timer = std::chrono::steady_clock::now();
        // During the run, update everything
        // is_done flag tracks whether all frames are processed
        std::shared_ptr<geometry::PointCloud> cur_frame_pcd, pre_frame_pcd;
        //pcl::PointCloud<pcl::PointXYZ>::Ptr  cur_frame_pcl_ptr, pre_frame_pcl_ptr;
        Eigen::Matrix4d pre_pose_gndtruth, pre_pose_icp, pre_pose_gmm, pre_pose_ndt, init_pose;

        // Files for storing the errors
        //ofstream icp_avg_file, icp_inst_file, gmm_avg_file, gmm_inst_file, ndt_avg_file, ndt_inst_file;
        //ndt_avg_file.open("ndt_avg.csv");
        //ndt_inst_file.open("ndt_inst.csv");

        // Files for storing the trajectory (in TUM format)
        std::ofstream gndtruth_traj_txt, icp_traj_txt, gmm_traj_txt, ndt_traj_txt;
        gndtruth_traj_txt.open("gndtruth_traj.txt");
        //ndt_traj_txt.open("ndt_traj.txt");

        // Initialize files and vectors for storing trajectory errors
        std::vector<double> result_avg_icp(6, 0.0);
        std::vector<double> result_avg_gmm(6, 0.0);
        std::vector<double> result_avg_ndt(6, 0.0);

        // Instantiate gmm_d2d object
        Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> T_gmmd2d;
        T_gmmd2d.setIdentity();

        // Track ground truth transformation
        Eigen::Transform<double,3,Eigen::Affine,Eigen::ColMajor> rel_gnd_trans, pre_gnd_trans;
        rel_gnd_trans.setIdentity();
        pre_gnd_trans.setIdentity();

        // Track states of some visualization parameters
        Eigen::Matrix4d T_eigen;
        gmm::map_param map_parameter;
        gmm::initializeMapParameters(map_parameter, dataset_name_);
        auto gmm_map = std::make_shared<gmm::GMMMap>(map_parameter,
                                                     &prop_values_.update_obs_gmms,
                                                     &prop_values_.update_free_gmms,
                                                     &prop_values_.fuse_gmm_across_frames);
        std::cout << fmt::format("Start GMM map construction with {} threads (cores). Tracking color: {}",
                                 gmm_map->mapParameters.num_threads, gmm_map->mapParameters.track_color) << std::endl;
        std::cout << fmt::format("GMM fusion with hell_thresh_squard_free: {:.4f}, hell_thresh_squard_obs: {:.4f}",
                                 gmm_map->mapParameters.hell_thresh_squard_free, gmm_map->mapParameters.hell_thresh_squard_obs) << std::endl;
        float unexplored_evidence = dataset_param::dataset_info["occupancy_inference_parameters"]["unexplored_evidence"].asFloat();
        float unexplored_variance = dataset_param::dataset_info["occupancy_inference_parameters"]["unexplored_variance"].asFloat();
        auto gmm_map_viz = new gmm::GMMMapViz(this->widget3d_, gmm_map, &prop_values_.surface_interval,
                                              kTangoGreen, OccVarColormapClass, OccVarColormapName,
                                              unexplored_evidence, unexplored_variance, prop_values_.ray_sampling_dist);

        bool final_frame = false;
        std::stringstream info, fps;
        info.setf(std::ios::fixed, std::ios::floatfield);
        info.precision(4);
        int frame_idx = idx + start_frame_idx;

        // Current viewing pivot
        Eigen::Vector3f viewing_pivot;
        Eigen::Vector3f cur_env_max_bound = env_bbox->GetMaxBound().cast<float>();
        Eigen::Vector3f cur_env_min_bound = env_bbox->GetMinBound().cast<float>();

        bool update_pcd = false;

        while (true) {
            if (!is_started_ || !is_running_ || is_done_) {
                final_frame = false;
                // If we aren't running, sleep a little bit so that we don't
                // use 100% of the CPU just checking if we need to run.
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                //std::cout << "Paused!" << std::endl;
                // Update the visibility of the geometries when completed!
                // The following is executed when paused!
                gui::Application::GetInstance().PostToMainThread(
                        this, [this, traj, frustum, coord_axes, gmm_map_viz, T_eigen, env_bbox, update_pcd,
                                info = info.str(), fps = fps.str()]() {
                            // Display axes
                            this->SetInfo(info);
                            this->SetFPS(fps);
                            this->widget3d_->GetScene()->ShowGeometry("coord_axes", prop_values_.show_coord);
                            //this->widget3d_->GetScene()->ShowGeometry("bbox", prop_values_.show_env_bbox);
                            if (is_started_) {
                                // Display and update trajectory
                                this->widget3d_->GetScene()->ShowGeometry("frustum", prop_values_.show_traj);

                                this->widget3d_->GetScene()->ShowGeometry("trajectory", prop_values_.show_traj);

                                gmm_map_viz->updateVisualizer(prop_values_.atomicMapVizFlags);

                                auto *scene = this->widget3d_->GetScene()->GetScene();

                                scene->ShowGeometry("points", prop_values_.show_pcd);

                                if (update_pcd) {
                                    using namespace rendering;
                                    auto mat_line = rendering::MaterialRecord();
                                    mat_line.shader = "unlitLine";
                                    mat_line.line_width = 5.0f;
                                    std::lock_guard<std::mutex> locker(surface_.lock);
                                    //std::cout << fmt::format("Number of points in the pointcloud: {}\n", surface_.pcd_cropped.ToLegacy().points_.size());
                                    if (surface_.pcd_cropped.HasPointPositions() &&
                                        surface_.pcd_cropped.HasPointColors()) {
                                        scene->UpdateGeometry(
                                                "points", surface_.pcd_cropped,
                                                Scene::kUpdatePointsFlag |
                                                Scene::kUpdateColorsFlag);
                                    }
                                    this->widget3d_->GetScene()->RemoveGeometry("bbox");
                                    this->widget3d_->GetScene()->AddGeometry("bbox", env_bbox.get(), mat_line);
                                }
                                this->widget3d_->GetScene()->ShowGeometry("bbox", prop_values_.show_env_bbox);
                            }
                        });
                //std::cout << "Done!" << std::endl;
                // Note: Do not update visualizer before the first frame is processed!
                if (is_terminated_){
                    return;
                } else if ((is_started_ && idx > 0) || is_done_) {
                    update_pcd = gmm_map_viz->isEnvBBoxUpdated(prop_values_.atomicMapVizFlags);
                    gmm_map_viz->updateGeometry(prop_values_.atomicMapVizFlags, final_frame);
                    if (update_pcd){
                        if (!gmm_map_viz->isMapCropped(prop_values_.atomicMapVizFlags)){
                            env_bbox->min_bound_ = cur_env_min_bound.cast<double>();
                            env_bbox->max_bound_ = cur_env_max_bound.cast<double>();
                            surface_.pcd_cropped = t::geometry::PointCloud::FromLegacy(surface_.pcd);
                        } else {
                            gmm_map_viz->adjustGlobalBBox(prop_values_.atomicMapVizFlags,
                                                          env_bbox->min_bound_, env_bbox->max_bound_);
                            surface_.pcd_cropped = open3d::t::geometry::PointCloud::FromLegacy(*surface_.pcd.Crop(*env_bbox));
                        }
                    }

                    updateInfoAndFPS(info, fps, frame_idx, rgb_files.size(), T_eigen, gmm_map_viz,
                                     1000.0 / (time_interval / total_fps_interval_len));
                }
                continue;
            }

            if (!is_done_) {
                if (idx == 0){
                    std::this_thread::sleep_for(std::chrono::milliseconds(std::max<int>(2000, prop_values_.sleep_ms)));
                } else {
                    std::this_thread::sleep_for(std::chrono::milliseconds(prop_values_.sleep_ms));
                }
                // Send the next image to the visualizer
                std::shared_ptr<t::geometry::Image> input_depth, input_color;
                //std::vector<std::vector<float>> depth_image;
                frame_idx = idx + start_frame_idx;
                if (frame_idx >= fmin(depth_files.size() - 1, start_frame_idx + num_frames - 1)){
                    final_frame = true;
                } else {
                    final_frame = false;
                }

                open3d::core::Tensor depth_tensor;
                gmm::RowMatrixXf depthmap;
                if (dataset_name_ == "tartanair") {
                    // Note that we should obtain the depth directly here to maximize overall throughput
                    input_depth = dutil::createImageNpy2O3d(depth_files[frame_idx], prop_values_.depth_scale);
                    depth_tensor = open3d::t::io::ReadNpy(depth_files[frame_idx]);
                    auto t_shape = depth_tensor.GetShape();
                    depthmap = open3d::core::eigen_converter::TensorToEigenMatrixXf(depth_tensor.Reshape({t_shape[0],t_shape[1]}));
                } else {
                    input_depth = t::io::CreateImageFromFile(depth_files[frame_idx]);
                    depth_tensor = input_depth->AsTensor();
                    auto t_shape = depth_tensor.GetShape();
                    if (dataset_name_ == "stata"){
                        gmm::RowMatrixXf depthmap_full = open3d::core::eigen_converter::TensorToEigenMatrixXf(depth_tensor.Reshape({t_shape[0],t_shape[1]}));
                        int stride = (int) (1.0 / dataset_param::sampling_rate);
                        depthmap = Eigen::Map<gmm::RowMatrixXf, 0, Stride<Dynamic, Dynamic>>(depthmap_full.data(),
                                                                                        depthmap_full.rows() / stride,
                                                                                        depthmap_full.cols() / stride,
                                                           Stride<Dynamic, Dynamic>(stride * depthmap_full.cols(), stride)) / dataset_param::scale;
                        *input_depth = input_depth->Resize(dataset_param::sampling_rate);
                    } else {
                        depthmap = open3d::core::eigen_converter::TensorToEigenMatrixXf(depth_tensor.Reshape({t_shape[0],t_shape[1]}))/dataset_param::scale;
                    }
                }

                input_color = t::io::CreateImageFromFile(rgb_files[frame_idx]);
                if (dataset_name_ == "stata"){
                    *input_color = input_color->Resize(dataset_param::sampling_rate);
                }

                input_frame.SetDataFromImage("depth", *input_depth);
                auto r_tensor = input_color->AsTensor().Slice(2,0,1);
                auto g_tensor = input_color->AsTensor().Slice(2,1,2);
                auto b_tensor = input_color->AsTensor().Slice(2,2,3);
                depth_tensor = input_depth->AsTensor();
                auto t_shape = depth_tensor.GetShape();
                //std::cout << "Shape of the depth channel is: " << depth_tensor.GetShape().ToString() << std::endl;
                //std::cout << "Shape of the rgb channel is: " << t_shape.ToString() << std::endl;
                gmm::RowMatrixXi r = open3d::core::eigen_converter::TensorToEigenMatrixXi(r_tensor.Reshape({t_shape[0],t_shape[1]})).cast<u_int8_t>();
                gmm::RowMatrixXi g = open3d::core::eigen_converter::TensorToEigenMatrixXi(g_tensor.Reshape({t_shape[0],t_shape[1]})).cast<u_int8_t>();
                gmm::RowMatrixXi b = open3d::core::eigen_converter::TensorToEigenMatrixXi(b_tensor.Reshape({t_shape[0],t_shape[1]})).cast<u_int8_t>();

                if (gmm_map->mapParameters.track_color && gmm_map->mapParameters.track_intensity){
                    gmm::RowMatrixXi input_intensity = (0.299f * r.cast<float>() + 0.587f * g.cast<float>() + 0.114f * b.cast<float>()).cast<uint8_t>();
                    open3d::core::Tensor IntensityRGB(input_color->AsTensor().GetShape(), input_color->GetDtype());
                    IntensityRGB.Slice(2,0,1) = core::eigen_converter::EigenMatrixToTensor(input_intensity).Reshape({t_shape[0],t_shape[1], 1});
                    IntensityRGB.Slice(2,1,2) = IntensityRGB.Slice(2,0,1);
                    IntensityRGB.Slice(2,2,3) = IntensityRGB.Slice(2,0,1);
                    input_frame.SetDataFromImage("color", IntensityRGB);
                } else {
                    input_frame.SetDataFromImage("color", *input_color);
                }

                auto rgbd_image = geometry::RGBDImage::CreateFromColorAndDepth(input_color->ToLegacy(), input_depth->ToLegacy(),
                                                                               prop_values_.depth_scale, prop_values_.depth_max);
                cur_frame_pcd = geometry::PointCloud::CreateFromRGBDImage(*rgbd_image, intrinsic_legacy);
                //cur_frame_pcd->UniformDownSample(10);

                // Create pcl object
                //cur_frame_pcl_ptr = convert::to_pcl<pcl::PointXYZ>(cur_frame_pcd);

                // Determine pose of the robot
                gmm::Isometry3 curPose;
                dutil::processPose(pose_files, frame_idx, dataset_name_,curPose);
                bool tracking_success = true;
                std::cout << "\nFrame number " << frame_idx+1 << "/" << depth_files.size() << std::endl;

                // Perform gmm construction
                if (depth_pixel_variance == 0){
                    gmm_map->insertFrame(r, g, b, depthmap,curPose);
                } else {
                    gmm::RowMatrixXf depth_variance(depthmap.rows(), depthmap.cols());
                    depth_variance.setConstant(depth_pixel_variance);
                    gmm_map->insertFrame(r, g, b, depthmap, depth_variance, curPose);
                }

                if (idx > 0) {
                    // Load ground truth pose
                    dutil::processPoseO3d(pose_files, frame_idx, dataset_name_, T_frame_to_model);
                    rel_gnd_trans.matrix() = pre_gnd_trans.matrix().inverse() * core::eigen_converter::TensorToEigenMatrixXd(T_frame_to_model);

                    // Print relative transformations:
                    //std::cout << "Ground Truth Pose:" << std::endl;
                    //std::cout << rel_gnd_trans.matrix() << std::endl;
                }

                //std::cout << "Pose matrix from the mapping: " << std::endl;
                //std::cout << core::eigen_converter::TensorToEigenMatrixXd(T_frame_to_model) << std::endl;

                gmm_map_viz->updateGeometry(prop_values_.atomicMapVizFlags, final_frame);

                // Transfer the pointcloud ownership
                pre_frame_pcd = cur_frame_pcd;
                //pre_frame_pcl_ptr = cur_frame_pcl_ptr;

                pre_gnd_trans.matrix() = core::eigen_converter::TensorToEigenMatrixXd(T_frame_to_model);

                // Integrate into VoxelHash
                model_->UpdateFramePose(idx, T_frame_to_model);
                if (tracking_success) {
                    model_->Integrate(input_frame, prop_values_.depth_scale,
                                      prop_values_.depth_max,
                                      prop_values_.trunc_voxel_multiplier);
                }

                // Update trajectory and camera frustrum
                T_eigen = core::eigen_converter::TensorToEigenMatrixXd(T_frame_to_model);
                if (traj->points_.empty() || traj->points_.back() != T_eigen.block<3, 1>(0, 3)) {
                    traj_param.extrinsic_ = T_eigen;
                    trajectory_->parameters_.push_back(traj_param);
                    traj->points_.emplace_back(T_eigen.block<3, 1>(0, 3));
                    if (traj->points_.size() > 1) {
                        int n = traj->points_.size();
                        traj->lines_.push_back({n - 1, n - 2});
                        traj->colors_.push_back(kTangoSkyBlueDark);
                    }
                }
                frustum = geometry::LineSet::CreateCameraVisualization(
                        color->width_, color->height_, K_eigen, T_eigen.inverse(),
                        0.2);
                frustum->PaintUniformColor(kTangoSkyBlueDark);

                // Record the actual trajectory on file
                Matrix3d rot = T_eigen.topLeftCorner(3,3);
                Quaterniond q(rot);
                gndtruth_traj_txt << fmt::format("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}", timestamps[frame_idx], T_eigen(0,3), T_eigen(1,3), T_eigen(2,3),
                                                 q.x(), q.y(), q.z(), q.w()) << std::endl;

                //std::shared_ptr<geometry::LineSet> gmm_linesets;
                pre_pose_gndtruth = T_eigen;

                // Update text display information
                if (idx % fps_interval_len == 0) {
                    total_fps_interval_len += fps_interval_len;
                    time_interval += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - timer).count();
                    timer = std::chrono::steady_clock::now();
                }

                updateInfoAndFPS(info, fps, frame_idx, rgb_files.size(), T_eigen, gmm_map_viz, 1000.0 / (time_interval / total_fps_interval_len));

                // TODO: update support for timages-image conversion
                color = std::make_shared<geometry::Image>(
                        input_frame.GetDataAsImage("color").ToLegacy());
                depth_colored = std::make_shared<geometry::Image>(
                        input_frame.GetDataAsImage("depth")
                                .ColorizeDepth(prop_values_.depth_scale, 0.001,
                                               prop_values_.depth_max)
                                .ToLegacy());

                // Extract surface on demand (do before we increment idx, so that
                // we see something immediately, on interaction 0)
                if ((prop_values_.update_surface &&
                     idx % static_cast<int>(prop_values_.surface_interval) == 0) ||
                        final_frame || !is_running_) {
                    std::lock_guard<std::mutex> locker(surface_.lock);
                    surface_.pcd_cropped =
                            model_->ExtractPointCloud(std::min<float>(idx, prop_values_.truncation_weight),
                                                      prop_values_.pointcloud_size)
                                    .To(core::Device("CPU:0"));
                    cur_env_max_bound = core::eigen_converter::TensorToEigenMatrixXf(surface_.pcd_cropped.GetMaxBound().Reshape({3,1}));
                    cur_env_min_bound = core::eigen_converter::TensorToEigenMatrixXf(surface_.pcd_cropped.GetMinBound().Reshape({3,1}));
                    env_bbox->min_bound_ = cur_env_min_bound.cast<double>();
                    env_bbox->max_bound_ = cur_env_max_bound.cast<double>();

                    if (!is_running_ || final_frame){
                        // Only copy if the pause button is detected!
                        surface_.pcd = surface_.pcd_cropped.ToLegacy();
                    }
                    is_scene_updated_ = true;
                }

                if (idx > 0){
                    // Average current viewing pivot
                    viewing_pivot = prop_values_.cur_viewpoint_weight * ((cur_env_max_bound + cur_env_min_bound) / 2 -
                                    Eigen::Vector3f{0,0,prop_values_.height_factor * invert_z * (cur_env_max_bound(2) - cur_env_min_bound(2)) / 2}) +
                                    (1.0f - prop_values_.cur_viewpoint_weight) * viewing_pivot;
                } else {
                    viewing_pivot = (cur_env_max_bound + cur_env_min_bound) / 2 -
                                    Eigen::Vector3f{0,0,prop_values_.height_factor * invert_z * (cur_env_max_bound(2) - cur_env_min_bound(2)) / 2};
                }

                // Start from prop_values_.zoom_factor. Zoom into the scene near the middle of the sequence.
                // Then, zoom back out to prop_values_.zoom_factor
                double adjusted_zoom = prop_values_.zoom_factor;
                double adjusted_zoom_low = 0.5;
                double adjusted_zoom_percentage = 0.4;
                if (adjusted_zoom > adjusted_zoom_low && prop_values_.enable_camera_tracking){
                    if (frame_idx + 1 < adjusted_zoom_percentage * depth_files.size()) {
                        adjusted_zoom = adjusted_zoom -
                                (adjusted_zoom - adjusted_zoom_low) *
                                (frame_idx + 1) / (adjusted_zoom_percentage * depth_files.size());
                    } else if (frame_idx + 1  > (1  - adjusted_zoom_percentage) * depth_files.size()) {
                        adjusted_zoom = adjusted_zoom_low +
                                (adjusted_zoom - adjusted_zoom_low) *
                                ((frame_idx + 1) - (1  - adjusted_zoom_percentage) * depth_files.size()) / (adjusted_zoom_percentage * depth_files.size());
                    } else {
                        adjusted_zoom = adjusted_zoom_low;
                    }
                    std::cout << fmt::format("Adjusted zoom factor: {:.2f}", adjusted_zoom) << std::endl;
                }

                // Update all geometries by posting updates to the main thread
                gui::Application::GetInstance().PostToMainThread(
                        this, [this, color, depth_colored, raycast_color,
                                raycast_depth_colored, traj, icp_traj, gmm_traj, gmm_map_viz, ndt_traj,
                                icp_frustum, gmm_frustum, ndt_frustum, frustum, coord_axes, env_bbox,
                                invert_z,
                                viewing_pivot,
                                cur_env_max_bound, cur_env_min_bound,
                                cur_pose = curPose.matrix().cast<float>(),
                                adjusted_zoom,
                                info = info.str(), fps = fps.str()]() {
                            // Disable depth_scale and pcd buffer size change
                            this->fixed_props_->SetEnabled(false);

                            this->SetInfo(info);
                            this->SetFPS(fps);
                            this->input_color_image_->UpdateImage(color);
                            this->input_depth_image_->UpdateImage(depth_colored);

                            // Display axes
                            this->widget3d_->GetScene()->ShowGeometry("coord_axes", prop_values_.show_coord);

                            auto mat_line = rendering::MaterialRecord();
                            mat_line.shader = "unlitLine";
                            mat_line.line_width = 5.0f;

                            // Display and update trajectory (gndTruth)
                            this->widget3d_->GetScene()->RemoveGeometry("frustum");
                            if (this->prop_values_.show_traj) {
                                this->widget3d_->GetScene()->AddGeometry(
                                        "frustum", frustum.get(), mat_line);
                            }

                            if (traj->points_.size() > 1) {
                                // 1) Add geometry once w/ max size
                                // 2) Update geometry
                                // TPointCloud
                                this->widget3d_->GetScene()->RemoveGeometry(
                                        "trajectory");
                                if (this->prop_values_.show_traj) {
                                    this->widget3d_->GetScene()->AddGeometry(
                                            "trajectory", traj.get(), mat_line);
                                }
                            }

                            // GMM Clusters
                            gmm_map_viz->updateVisualizer(prop_values_.atomicMapVizFlags);

                            // Sampled points
                            auto mat = rendering::MaterialRecord();
                            mat.point_size = 3.0f;
                            //this->widget3d_->GetScene()->RemoveGeometry("free_pcd");
                            //this->widget3d_->GetScene()->AddGeometry("free_pcd", free_pcd.get(), mat);
                            //this->widget3d_->GetScene()->ShowGeometry("free_pcd", prop_values_.show_free_pcd);
                            //this->widget3d_->GetScene()->RemoveGeometry("obs_pcd");
                            //this->widget3d_->GetScene()->AddGeometry("obs_pcd", obs_pcd.get(), mat);
                            //this->widget3d_->GetScene()->ShowGeometry("obs_pcd", prop_values_.show_obs_pcd);

                            // Voxels
                            auto *scene = this->widget3d_->GetScene()->GetScene();
                            if (is_scene_updated_) {
                                using namespace rendering;
                                std::lock_guard<std::mutex> locker(surface_.lock);
                                if (surface_.pcd_cropped.HasPointPositions() &&
                                    surface_.pcd_cropped.HasPointColors()) {
                                    scene->UpdateGeometry(
                                            "points", surface_.pcd_cropped,
                                            Scene::kUpdatePointsFlag |
                                            Scene::kUpdateColorsFlag);
                                }

                                this->widget3d_->GetScene()->RemoveGeometry("bbox");
                                this->widget3d_->GetScene()->AddGeometry("bbox", env_bbox.get(), mat_line);
                                is_scene_updated_ = false;
                            }
                            scene->ShowGeometry("points", prop_values_.show_pcd);
                            this->widget3d_->GetScene()->ShowGeometry("bbox", prop_values_.show_env_bbox);

                            // Set view using the pose of the robot
                            if (prop_values_.enable_camera_tracking){
                                Eigen::Vector3f forward = (viewing_pivot - cur_pose.topRightCorner<3,1>().cast<float>()).normalized();
                                float dist = adjusted_zoom * std::fmax(cur_env_max_bound(2) - cur_env_min_bound(2) , 10.0f);
                                this->widget3d_->LookAt(cur_pose.topRightCorner<3,1>().cast<float>(),
                                                        cur_pose.topRightCorner<3,1>().cast<float>() - dist * forward,
                                                        Eigen::Vector3f{0,0,invert_z * 1});
                            }

                        });

                // Note that the user might have closed the window, in which case we
                // want to maintain a value of true.
                idx++;
                is_done_ = is_done_ | final_frame;
                std::cout << gmm_map->printStatistics(frame_idx, frame_idx, false) << std::endl;
                if (is_done_){
                    std::cout << gmm_map->printStatistics(frame_idx+1, frame_idx+1) << std::endl;
                    std::cout << "Sequence completed!\n" << std::endl;

                    if (prop_values_.save_map){
                        std::filesystem::create_directories(dataset_param::result_path);
                        std::string map_file = dataset_param::result_path / "gmmap.bin";
                        std::cout << fmt::format("Saving map to a binary file: {}\n", map_file);
                        std::ofstream stream(map_file, std::ios::out | std::ios::binary);
                        if(!stream) {
                            std::cout << "Cannot open file for writing!\n" << std::endl;
                        } else {
                            auto save_start = std::chrono::steady_clock::now();
                            gmm_map->save(stream);
                            long save_duration = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - save_start).count();
                            std::cout << fmt::format("Map file successfully saved in {}us\n", save_duration);
                        }
                        stream.close();

                        std::string pcd_file = dataset_param::result_path / "pointcloud.pcd";
                        std::cout << fmt::format("Saving pointcloud to file: {}\n", pcd_file);
                        open3d::io::WritePointCloudOption write_params;
                        open3d::io::WritePointCloudToPCD(pcd_file, surface_.pcd, write_params);
                        std::cout << "Completed saving pointcloud!\n\n";
                    }

                    if (prop_values_.save_occ_img){
                        gmm::V box_min = gmm_map_viz->map_BBox.min();
                        gmm::V box_max =  gmm_map_viz->map_BBox.max();
                        gmm::V box_extent = box_max - box_min;
                        for (int i = 0; i < 10; i++){
                            gmm::FP height = box_min(2) + 0.1f * (gmm::FP) i * box_extent(2);
                            std::string occ_filename = fmt::format("occupancy_height_{}", i);
                            gmm_map_viz->generateCrossSectionOccupancyMap(2,height,occ_filename,1000);
                        }
                    }
                }
            }

        }
    }
};

//------------------------------------------------------------------------------
// Print usage informations
void PrintHelp() {
    using namespace open3d;

    PrintOpen3DVersion();
    // clang-format off
    utility::LogInfo("Usage:");
    utility::LogInfo("    > VoxelHashingGUI [dataset name]");
    utility::LogInfo("      Given a sequence of RGBD images, reconstruct point cloud from color and depth images");
    utility::LogInfo("");
    utility::LogInfo("Basic options:");
    utility::LogInfo("    --voxel_size [=0.0058 (m)]");
    utility::LogInfo("    --device [CUDA:0]");
    // clang-format on
    utility::LogInfo("");
}

int main(int argc, char* argv[]) {
    // argc = arguement count
    // argv = arguement vector
    using namespace open3d;

    if (argc < 8 || utility::ProgramOptionExistsAny(argc, argv, {"-h", "--help"})) {
        std::cout << "Input format: dataset dataset_info_path start_frame_idx num_frames depth_pixel_variance --device [CPU:0 or CUDA:0]" <<std::endl;
        return 1;
    }

    std::string dataset_name = argv[1];
    std::string dataset_config_file_path = argv[2];
    int start_frame_idx = atoi(argv[3]);
    int num_frames = atoi(argv[4]);
    float depth_pixel_variance = atof(argv[5]);
    dutil::loadDatasetInfoFromJSON(dataset_config_file_path, dataset_name);

    //std::string intrinsic_path = utility::GetProgramOptionAsString(
    //        argc, argv, "--intrinsics_json", "");

    std::string device_code =
            utility::GetProgramOptionAsString(argc, argv, "--device", "CUDA:0");
    if (device_code != "CPU:0" && device_code != "CUDA:0") {
        utility::LogWarning(
                "Unrecognized device {}. Expecting CPU:0 or CUDA:0.",
                device_code);
        return -1;
    }
    utility::LogInfo("Using device {}.", device_code);

    auto& app = gui::Application::GetInstance();
    app.Initialize(argc, const_cast<const char**>(argv));
    auto mono =
            app.AddFont(gui::FontDescription(gui::FontDescription::MONOSPACE));
    app.AddWindow(std::make_shared<ReconstructionWindow>(dataset_name,
                                                         start_frame_idx,
                                                         num_frames,
                                                         depth_pixel_variance,
                                                         device_code, mono));
    app.Run();
}
