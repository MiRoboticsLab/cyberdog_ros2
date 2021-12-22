// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef MANAGERS__MOTION_MANAGER_HPP_
#define MANAGERS__MOTION_MANAGER_HPP_

// C++ headers
#include <chrono>
#include <condition_variable>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fstream>

// C++17 headers not support uncrustify yet
#include "string_view"

// automation msgs
#include "automation_msgs/msg/caution.hpp"
#include "automation_msgs/srv/nav_mode.hpp"
#include "nav_msgs/msg/odometry.hpp"
// cacadenode msgs
#include "cascade_lifecycle_msgs/msg/activation.hpp"
#include "cascade_lifecycle_msgs/msg/state.hpp"
// ception msgs
#include "ception_msgs/msg/around.hpp"
#include "ception_msgs/srv/sensor_detection_node.hpp"
// lcm msgs
#include "lcm_translate_msgs/motion_control_request_lcmt.hpp"
#include "lcm_translate_msgs/motion_control_response_lcmt.hpp"
#include "lcm_translate_msgs/state_estimator_lcmt.hpp"
#include "lcm_translate_msgs/trajectory_command_lcmt.hpp"
// motion msgs
#include "motion_msgs/msg/frameid.hpp"
#include "motion_msgs/msg/se3_pose.hpp"
#include "motion_msgs/msg/se3_velocity.hpp"
#include "motion_msgs/msg/se3_velocity_cmd.hpp"
#include "motion_msgs/msg/mode.hpp"
#include "motion_msgs/msg/parameters.hpp"
#include "motion_msgs/msg/gait.hpp"
#include "motion_msgs/msg/safety.hpp"
#include "motion_msgs/msg/scene.hpp"
#include "motion_msgs/msg/control_state.hpp"
#include "motion_msgs/msg/error_flag.hpp"
#include "motion_msgs/action/change_mode.hpp"
#include "motion_msgs/action/change_gait.hpp"
#include "motion_msgs/action/ext_mon_order.hpp"
// ROS headers
#include "ament_index_cpp/get_package_share_directory.hpp"
#include "motion_bridge/gait_interface.hpp"
#include "cyberdog_utils/Enums.hpp"
#include "cyberdog_utils/action_server.hpp"
#include "manager_utils/bt_action_server.hpp"
#include "manager_utils/cascade_manager.hpp"
#include "managers/automation_manager.hpp"
#include "rclcpp/rclcpp.hpp"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/utils.h"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/create_timer_ros.h"
#include "tf2_ros/transform_broadcaster.h"
// other headers
#include "lcm/lcm-cpp.hpp"
#include "toml11/toml.hpp"

namespace cyberdog
{
namespace manager
{

// interfaces
using Around_T = ception_msgs::msg::Around;
using CallbackReturn_T = cyberdog_utils::CallbackReturn;
using CascadeActivation_T = cascade_lifecycle_msgs::msg::Activation;
using CascadeState_T = cascade_lifecycle_msgs::msg::State;
using ChangeMode_T = motion_msgs::action::ChangeMode;
using ChangeGait_T = motion_msgs::action::ChangeGait;
using ControlState_T = motion_msgs::msg::ControlState;
using ExtMonOrder_T = motion_msgs::action::ExtMonOrder;
using FrameID_T = motion_msgs::msg::Frameid;
using Gait_T = motion_msgs::msg::Gait;
using GaitFB_T = motion_msgs::action::ChangeGait_Feedback;
using GaitRes_T = motion_msgs::action::ChangeGait_Result;
using GoalHandleGait_T = rclcpp_action::ClientGoalHandle<ChangeGait_T>;
using LifecycleNode_T = cyberdog_utils::LifecycleNode;
using Mode_T = motion_msgs::msg::Mode;
using ModeFB_T = motion_msgs::action::ChangeMode_Feedback;
using ModeRes_T = motion_msgs::action::ChangeMode_Result;
using MonOrder_T = motion_msgs::msg::MonOrder;
using MonOrderFB_T = motion_msgs::action::ExtMonOrder_Feedback;
using MonOrderRes_T = motion_msgs::action::ExtMonOrder_Result;
using NavCaution_T = automation_msgs::msg::Caution;
using Parameters_T = motion_msgs::msg::Parameters;
using Quaternion_T = geometry_msgs::msg::Quaternion;
using Safety_T = motion_msgs::msg::Safety;
using Scene_T = motion_msgs::msg::Scene;
using SE3Pose_T = motion_msgs::msg::SE3Pose;
using SE3Velocity_T = motion_msgs::msg::SE3Velocity;
using SE3VelocityCMD_T = motion_msgs::msg::SE3VelocityCMD;
using Time_T = builtin_interfaces::msg::Time;
using SubMode_T = automation_msgs::srv::NavMode;
using SubModeReq_T = automation_msgs::srv::NavMode::Request;
using ErrorFlag_T = motion_msgs::msg::ErrorFlag;

// enums
using GaitChangePriority_T = cyberdog_utils::GaitChangePriority;

enum INTER_CHANGE_TYPE
{
  DEFAULT = 0,
  SUCCEED = 1,
  FAILED = 2
};

class MotionManager : public manager::CascadeManager
{
public:
  MotionManager();
  ~MotionManager();

protected:
  CallbackReturn_T on_configure(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_activate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_deactivate(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_cleanup(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_shutdown(const rclcpp_lifecycle::State &) override;
  CallbackReturn_T on_error(const rclcpp_lifecycle::State &) override;

private:
// Callback group
  rclcpp::CallbackGroup::SharedPtr callback_group_service;

// Math function  *to be moved together in future
  template<typename T>
  T limit_data(T in, T min, T max);

// Action servers callback
  void checkout_mode();
  void checkout_gait();
  void mon_order_exec();

// Topic subscription callback
  void velocity_cmd_callback(const SE3VelocityCMD_T::SharedPtr msg);
  void ob_detection_callback(const Around_T::SharedPtr msg);
  void paras_callback(const Parameters_T::SharedPtr msg);
  void caution_callback(const NavCaution_T::SharedPtr msg);
  void control_state_callback(const ControlState_T::SharedPtr msg);
  void guard_callback(const Safety_T::SharedPtr msg);

// LCM message callback & LCM handler
  void control_lcm_collection(
    const lcm::ReceiveBuffer * rbuf,
    const std::string & channel,
    const motion_control_response_lcmt * lcm_data);
  void statees_lcm_collection(
    const lcm::ReceiveBuffer * rbuf,
    const std::string & channel,
    const state_estimator_lcmt * lcm_data);
  void recv_lcm_control_handle();
  void recv_lcm_statees_handle();
  inline std::string get_lcm_url(std::string ip, int port, int ttl)
  {
    return std::string("udpm://") +
           ip + std::string(":" + std::to_string(port) + "?ttl=") + std::to_string(ttl);
  }

// Topic publisher
  void publish_velocity(
    const SE3VelocityCMD_T::SharedPtr velocity_out,
    const int8_t ORDER_TYPE = MonOrder_T::MONO_ORDER_NULL);
  void publish_mode(const Mode_T & mode_out);
  void publish_gait(const Gait_T & gait_out, const bool order_req = false);
  void publish_control_state(const ControlState_T & control_state_out);
  void publish_paras(const Parameters_T paras_out);
  void publish_odom(const nav_msgs::msg::Odometry & odom_out);

// Internal abstract function
  void init();
  std_msgs::msg::Header return_custom_header(std::string frame_id);
  bool check_time_update(
    Time_T coming_time,
    Time_T stash_time);

// Internal function
  void movement_detection();
  // Only for gait checking in action server
  bool inter_check_gait_sync(
    const Gait_T & goal_gait, const uint32_t & timeout,
    const uint8_t & priority);
  void automation_node_spin();
  void control_cmd_spin();
  template<typename T>
  void parameter_check(
    const T & max_value, T & value_to_check, const std::string & value_name,
    const bool & max_check = true);
  void reset_velocity(const uint8_t & order_id = 0);
  bool check_motor_errflag(bool show_all = false);
  bool toml_to_lcm(const toml::table & step, trajectory_command_lcmt & traj_cmd);

/// Variables
// parameters<string> topic_name
  std::unordered_map<std::string, std::string> topic_name_map_;
// parameters<int> rate
  inline static int rate_common_;
  inline static int rate_control_;
  inline static int rate_output_;
  inline static int rate_wait_loop_;
  inline static int rate_odom_;
  inline static int rate_lcm_const_;
// parameters<int> timeout
  inline static int timeout_manager_;
  inline static int timeout_motion_;
  inline static int timeout_gait_;
  inline static int timeout_order_;
  inline static int timeout_lcm_;
// parameters<double> constraint
  inline static double cons_abs_lin_x_;
  inline static double cons_abs_lin_y_;
  inline static double cons_abs_ang_r_;
  inline static double cons_abs_ang_p_;
  inline static double cons_abs_ang_y_;
  inline static double cons_abs_aang_y_;
  inline static double cons_max_body_;
  inline static double cons_max_gait_;
  inline static double cons_default_body_;
  inline static double cons_default_gait_;
  inline static double cons_speed_l_normal_;
  inline static double cons_speed_a_normal_;
// parameters<double> scale
  inline static double scale_low_btr_;
// parameters<int> ttl_config
  inline static int ttl_recv_from_motion_;
  inline static int ttl_send_to_motion_;
  inline static int ttl_from_odom_;

// parameters<int> ports_config
  inline static int port_recv_from_motion_;
  inline static int port_send_to_motion_;
  inline static int port_from_odom_;
// Transforms
  std::unique_ptr<tf2_ros::Buffer> tf_;
  std::unique_ptr<tf2_ros::TransformListener> tf_listener_;
  geometry_msgs::msg::TransformStamped robot_body_tf_;
  geometry_msgs::msg::TransformStamped robot_global_tf_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster;

// LCM handler & messages
  std::unique_ptr<lcm::LCM> motion_out_;
  std::unique_ptr<lcm::LCM> motion_in_;
  std::unique_ptr<lcm::LCM> state_es_in_;
  inline static motion_control_request_lcmt ros_to_lcm_data_;
  std::vector<trajectory_command_lcmt> _motionList;
  inline static uint32_t response_count_;
  inline static uint32_t odom_count_;

// Internal variables
// Gait & locomotion variables
  bool thread_flag_;
  Gait_T gait_cached_;
  Time_T last_motion_time_;
  bridge::GaitInterface gait_interface_;
// Robot state variables
  Around_T obstacle_data_;
  ControlState_T robot_control_state_;
  nav_msgs::msg::Odometry odom_;
  NavCaution_T nav_caution_;
  SE3VelocityCMD_T ext_velocity_cmd_;
// Order variables
  uint8_t order_running_;
  int8_t tqdm_single_;
  int8_t tqdm_multi_;
// Package directories
  std::string local_params_dir;
  std::string locomotion_params_dir;

// Node Executors
  rclcpp::executors::SingleThreadedExecutor node_exec_;

// Threads ptr
  std::unique_ptr<std::thread> lcm_control_res_handle_thread_;
  std::unique_ptr<std::thread> lcm_statees_res_handle_thread_;
  std::unique_ptr<std::thread> automation_node_thread_;
  std::unique_ptr<std::thread> control_cmd_thread_;
  std::unique_ptr<std::thread> ros_switch_order_thread_;

// Action Server
  std::unique_ptr<cyberdog_utils::ActionServer<ChangeMode_T, LifecycleNode_T>> mode_server_;
  std::unique_ptr<cyberdog_utils::ActionServer<ChangeGait_T,
    LifecycleNode_T>> gait_server_;
  std::unique_ptr<cyberdog_utils::ActionServer<ExtMonOrder_T,
    LifecycleNode_T>> monorder_server_;
  std::unique_ptr<bt_engine::BtActionServer<ChangeGait_T,
    LifecycleNode_T>> gait_server_test_;

// Action Client
  rclcpp_action::Client<ChangeGait_T>::SharedPtr gait_client_;

// Service Server
//
// Service Client
  rclcpp::Client<automation_msgs::srv::NavMode>::SharedPtr casual_service_client_;
  rclcpp::Client<ception_msgs::srv::SensorDetectionNode>::SharedPtr ob_detect_client_;

// Subscriber
  rclcpp::Subscription<SE3VelocityCMD_T>::SharedPtr velocity_sub_;
  rclcpp::Subscription<Around_T>::SharedPtr ob_detect_sub_;
  rclcpp::Subscription<Parameters_T>::SharedPtr paras_sub_;
  rclcpp::Subscription<NavCaution_T>::SharedPtr cau_sub_;
  rclcpp::Subscription<ControlState_T>::SharedPtr control_state_sub_;
  rclcpp::Subscription<Safety_T>::SharedPtr guard_sub_;

// Publisher
  rclcpp_lifecycle::LifecyclePublisher<Gait_T>::SharedPtr gait_pub_;
  rclcpp::Publisher<Gait_T>::SharedPtr gait_pub_temp_;
  rclcpp_lifecycle::LifecyclePublisher<ControlState_T>::SharedPtr control_state_pub_;
  rclcpp_lifecycle::LifecyclePublisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  rclcpp_lifecycle::LifecyclePublisher<SE3VelocityCMD_T>::SharedPtr cmd_pub_;

// Node individual
  std::shared_ptr<manager::AutomationManager> automation_manager_node_;

// automationmanager node list
  std::unordered_map<uint8_t, std::string> auto_node_list_
  {
    {Mode_T::MODE_EXPLOR, std::string("explor_nodes_l")},
    {Mode_T::MODE_TRACK, std::string("tracking_nodes_l")}
  };
// Log maps
  std::unordered_map<uint8_t, std::string> mode_label_
  {
    {Mode_T::MODE_DEFAULT, std::string("mode_default")},
    {Mode_T::MODE_LOCK, std::string("mode_lock")},
    {Mode_T::MODE_SEMI, std::string("mode_semi")},
    {Mode_T::MODE_EXPLOR, std::string("mode_explor")},
    {Mode_T::MODE_TRACK, std::string("mode_track")},
    {Mode_T::MODE_MANUAL, std::string("mode_manual")}
  };
  std::unordered_map<uint8_t, std::string> mode_errcode_label_
  {
    {ModeRes_T::NORMAL, std::string("null")},
    {ModeRes_T::FAILED, std::string("failed")},
    {ModeRes_T::REJECT, std::string("reject")},
    {ModeRes_T::CANCELED, std::string("canceled")},
    {ModeRes_T::BAD_TIMESTAMP, std::string("bad_timestamp")},
    {ModeRes_T::UNKNOWN, std::string("unknown")},
    {ModeRes_T::TIME_OUT, std::string("timeout")},
    {ModeRes_T::UNAVAILABLE, std::string("unavailable")},
    {ModeRes_T::AVOID_PREEMPT, std::string("avoid_preempt")}
  };
  std::unordered_map<uint8_t, std::string> mode_errstate_label_
  {
    {ModeFB_T::NORMAL_STATE, std::string("normal")},
    {ModeFB_T::CHECKING_GAIT, std::string("checking_gait")},
    {ModeFB_T::WAITING_NODES, std::string("waiting_nodes_up")},
    {ModeFB_T::CHECKING_SUBMODE, std::string("checking_submode")}
  };
  std::unordered_map<uint8_t, std::string> submode_label_
  {
    {Mode_T::MODE_DEFAULT, std::string("submode_null")},
    {Mode_T::TRACK_F, std::string("submode_trackface")},
    {Mode_T::TRACK_S, std::string("submode_trackselect")},
    {Mode_T::EXPLR_NAV_AB, std::string("submode_nav_ab")},
    {Mode_T::EXPLR_MAP_U, std::string("submode_map_update")},
    {Mode_T::EXPLR_MAP_N, std::string("submode_map_new")}
  };
  std::unordered_map<uint8_t, std::string> gait_label_
  {
    {Gait_T::GAIT_TRANS, std::string("trans")},
    {Gait_T::GAIT_DEFAULT, std::string("default")},
    {Gait_T::GAIT_PASSIVE, std::string("passive")},
    {Gait_T::GAIT_KNEEL, std::string("kneel")},
    {Gait_T::GAIT_STAND_R, std::string("recovery_stand")},
    {Gait_T::GAIT_STAND_B, std::string("qp_stand")},
    {Gait_T::GAIT_AMBLE, std::string("amble")},
    {Gait_T::GAIT_WALK, std::string("walk")},
    {Gait_T::GAIT_SLOW_TROT, std::string("slowtrot")},
    {Gait_T::GAIT_TROT, std::string("trot")},
    {Gait_T::GAIT_FLYTROT, std::string("flytrot")},
    {Gait_T::GAIT_BOUND, std::string("bound")},
    {Gait_T::GAIT_PRONK, std::string("pronk")}
  };
  std::unordered_map<uint8_t, std::string> gait_errcode_label_
  {
    {GaitRes_T::NORMAL, std::string("null")},
    {GaitRes_T::FAILED, std::string("failed")},
    {GaitRes_T::REJECT, std::string("reject")},
    {GaitRes_T::CANCELED, std::string("canceled")},
    {GaitRes_T::BAD_TIMESTAMP, std::string("bad_timestamp")},
    {GaitRes_T::UNKNOWN, std::string("unknown")},
    {GaitRes_T::TIME_OUT, std::string("timeout")},
    {GaitRes_T::UNAVAILABLE, std::string("unavailable")},
    {GaitRes_T::STUCK, std::string("stuck")},
    {GaitRes_T::AVOID_PREEMPT, std::string("avoid_preempt")}
  };
  std::unordered_map<uint8_t, std::string> gait_motivation_map_
  {
    {cyberdog_utils::LOCK_DETECT, std::string("lock_detect")},
    {cyberdog_utils::MODE_TRIG, std::string("mode_trig")},
    {cyberdog_utils::GAIT_TRIG, std::string("pattern_trig")},
    {cyberdog_utils::ORDER_REQ, std::string("order_req")}
  };
  std::unordered_map<uint8_t, std::string> source_id_map_
  {
    {SE3VelocityCMD_T::INTERNAL, std::string("internal")},
    {SE3VelocityCMD_T::REMOTEC, std::string("remote_controller")},
    {SE3VelocityCMD_T::NAVIGATOR, std::string("navigator")}
  };
  std::map<std::pair<uint8_t, uint8_t>, uint8_t> submode_map_
  {
    {{Mode_T::MODE_EXPLOR, Mode_T::EXPLR_MAP_U}, SubModeReq_T::EXPLR_MAP_UPDATE},
    {{Mode_T::MODE_EXPLOR, Mode_T::EXPLR_MAP_N}, SubModeReq_T::EXPLR_MAP_NEW},
    {{Mode_T::MODE_EXPLOR, Mode_T::EXPLR_NAV_AB}, SubModeReq_T::EXPLR_NAV_AB},
    {{Mode_T::MODE_TRACK, Mode_T::TRACK_F}, SubModeReq_T::TRACK_F},
    {{Mode_T::MODE_TRACK, Mode_T::TRACK_S}, SubModeReq_T::TRACK_S}
  };
  std::unordered_map<uint8_t, std::string> order_label_
  {
    {MonOrder_T::MONO_ORDER_NULL, std::string("null")},
    {MonOrder_T::MONO_ORDER_STAND_UP, std::string("stand_up")},
    {MonOrder_T::MONO_ORDER_PROSTRATE, std::string("prostrate")},
    {MonOrder_T::MONO_ORDER_STEP_BACK, std::string("step_back")},
    {MonOrder_T::MONO_ORDER_TURN_AROUND, std::string("turn_around")},
    {MonOrder_T::MONO_ORDER_HI_FIVE, std::string("hi_five")},
    {MonOrder_T::MONO_ORDER_DANCE, std::string("dance")},
    {MonOrder_T::MONO_ORDER_WELCOME, std::string("welcome")},
    {MonOrder_T::MONO_ORDER_TURN_OVER, std::string("turn_over")},
    {MonOrder_T::MONO_ORDER_SIT, std::string("sit")},
    {MonOrder_T::MONO_ORDER_SHOW, std::string("show")}
  };
  std::unordered_map<uint8_t, std::string> order_errorcode_label_
  {
    {MonOrderRes_T::NORMAL, std::string("normal")},
    {MonOrderRes_T::FAILED, std::string("failed")},
    {MonOrderRes_T::REJECT, std::string("reject")},
    {MonOrderRes_T::CANCELED, std::string("canceled")},
    {MonOrderRes_T::BAD_TIMESTAMP, std::string("bad_timestamp")},
    {MonOrderRes_T::UNKNOWN, std::string("unknown")},
    {MonOrderRes_T::TIME_OUT, std::string("timeout")},
    {MonOrderRes_T::UNAVAILABLE, std::string("unavailable")},
    {MonOrderRes_T::AVOID_PREEMPT, std::string("avoid_preempt")},
    {MonOrderRes_T::INTERRUPTED, std::string("interrupted")},
    {MonOrderRes_T::FILE_MISSED, std::string("file_missed")},
    {MonOrderRes_T::STUCK, std::string("stuck")}
  };
  std::unordered_map<uint8_t, uint8_t> order_to_gait_
  {
    {MonOrder_T::MONO_ORDER_STAND_UP, Gait_T::GAIT_STAND_R},
    {MonOrder_T::MONO_ORDER_PROSTRATE, Gait_T::GAIT_KNEEL},
    {MonOrder_T::MONO_ORDER_STEP_BACK, Gait_T::GAIT_TROT},
    {MonOrder_T::MONO_ORDER_TURN_AROUND, Gait_T::GAIT_TROT}
  };
};  // class MotionManager
}  // namespace manager
}  // namespace cyberdog

#endif  // MANAGERS__MOTION_MANAGER_HPP_
