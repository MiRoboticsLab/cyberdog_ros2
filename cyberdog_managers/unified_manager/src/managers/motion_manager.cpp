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

#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <experimental/filesystem>  // NOLINT
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "managers/motion_manager.hpp"

#define LOCOMOTION_BRIDGE_PKG "cyberdog_motion_bridge"

namespace cyberdog
{
namespace manager
{

MotionManager::MotionManager()
: manager::CascadeManager("motion_manager", "")
{
  message_info(std::string("Creating ") + this->get_name());

  // Get package share dir
  #ifdef PACKAGE_NAME
  auto local_share_dir = ament_index_cpp::get_package_share_directory(PACKAGE_NAME);
  local_params_dir = local_share_dir + std::string("/params/");
  #endif
  auto locomotion_share_dir = ament_index_cpp::get_package_share_directory(LOCOMOTION_BRIDGE_PKG);
  locomotion_params_dir = locomotion_share_dir + std::string("/params/");

  // Declare this node's parameters
  this->declare_parameter("topic_name_rc", "body_cmd");
  this->declare_parameter("rate_common_hz", 30);
  this->declare_parameter("rate_control_hz", 30);
  this->declare_parameter("rate_output_hz", 5);
  this->declare_parameter("rate_wait_loop_hz", 2);
  this->declare_parameter("rate_odom_hz", 50);
  this->declare_parameter("rate_lcm_const_hz", 500);
  this->declare_parameter("timeout_manager_s", 10);
  this->declare_parameter("timeout_motion_ms", 300);
  this->declare_parameter("timeout_gait_s", 12);
  this->declare_parameter("timeout_order_ms", 500);
  this->declare_parameter("timeout_lcm_ms", 200);
  this->declare_parameter("cons_abs_lin_x_mps", 1.6);
  this->declare_parameter("cons_abs_lin_y_mps", 0.8);
  this->declare_parameter("cons_abs_ang_r_rps", 0.5);
  this->declare_parameter("cons_abs_ang_p_rps", 0.5);
  this->declare_parameter("cons_abs_ang_y_rps", 1.0);
  this->declare_parameter("cons_abs_aang_y_rps2", 1.0);
  this->declare_parameter("cons_max_body_m", 0.5);
  this->declare_parameter("cons_max_gait_m", 0.2);
  this->declare_parameter("cons_default_body_m", 0.3);
  this->declare_parameter("cons_default_gait_m", 0.08);
  this->declare_parameter("cons_speed_a_normal_rps", 1.0);
  this->declare_parameter("cons_speed_l_normal_mps", 0.3);
  this->declare_parameter("scale_low_btr_r", 0.8);
  this->declare_parameter("ttl_recv_from_motion", 12);
  this->declare_parameter("ttl_send_to_motion", 2);
  this->declare_parameter("ttl_from_odom", 255);
  this->declare_parameter("port_recv_from_motion", 7670);
  this->declare_parameter("port_send_to_motion", 7671);
  this->declare_parameter("port_from_odom", 7669);
  message_info(this->get_name() + std::string(" created"));
}

MotionManager::~MotionManager()
{
  if (get_current_state().id() ==
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
  {
    on_deactivate(get_current_state());
    on_cleanup(get_current_state());
  }
  message_info(this->get_name() + std::string(" lifecycle destroyed"));
}

CallbackReturn_T MotionManager::on_configure(const rclcpp_lifecycle::State &)
{
  message_info(this->get_name() + std::string(" onconfiguring"));

  topic_name_map_.insert(
    std::pair<std::string, std::string>(
      "rc_topic",
      get_parameter("topic_name_rc").as_string()));
  rate_common_ = get_parameter("rate_common_hz").as_int();
  rate_control_ = get_parameter("rate_control_hz").as_int();
  rate_output_ = get_parameter("rate_output_hz").as_int();
  rate_wait_loop_ = get_parameter("rate_wait_loop_hz").as_int();
  rate_odom_ = get_parameter("rate_odom_hz").as_int();
  rate_lcm_const_ = get_parameter("rate_lcm_const_hz").as_int();
  timeout_manager_ = get_parameter("timeout_manager_s").as_int();
  timeout_motion_ = get_parameter("timeout_motion_ms").as_int();
  timeout_gait_ = get_parameter("timeout_gait_s").as_int();
  timeout_order_ = get_parameter("timeout_order_ms").as_int();
  timeout_lcm_ = get_parameter("timeout_lcm_ms").as_int();
  cons_abs_lin_x_ = get_parameter("cons_abs_lin_x_mps").as_double();
  cons_abs_lin_y_ = get_parameter("cons_abs_lin_y_mps").as_double();
  cons_abs_ang_r_ = get_parameter("cons_abs_ang_r_rps").as_double();
  cons_abs_ang_p_ = get_parameter("cons_abs_ang_p_rps").as_double();
  cons_abs_ang_y_ = get_parameter("cons_abs_ang_y_rps").as_double();
  cons_abs_aang_y_ = get_parameter("cons_abs_aang_y_rps2").as_double();
  cons_max_body_ = get_parameter("cons_max_body_m").as_double();
  cons_max_gait_ = get_parameter("cons_max_gait_m").as_double();
  cons_default_body_ = get_parameter("cons_default_body_m").as_double();
  cons_default_gait_ = get_parameter("cons_default_gait_m").as_double();
  cons_speed_a_normal_ = get_parameter("cons_speed_a_normal_rps").as_double();
  cons_speed_l_normal_ = get_parameter("cons_speed_l_normal_mps").as_double();
  scale_low_btr_ = get_parameter("scale_low_btr_r").as_double();
  ttl_recv_from_motion_ = get_parameter("ttl_recv_from_motion").as_int();
  ttl_send_to_motion_ = get_parameter("ttl_send_to_motion").as_int();
  ttl_from_odom_ = get_parameter("ttl_from_odom").as_int();
  port_recv_from_motion_ = get_parameter("port_recv_from_motion").as_int();
  port_send_to_motion_ = get_parameter("port_send_to_motion").as_int();
  port_from_odom_ = get_parameter("port_from_odom").as_int();

  parameter_check(rate_lcm_const_, rate_odom_, std::string("rate_odom"));
  parameter_check(rate_lcm_const_, rate_output_, std::string("rate_output"));
  parameter_check(rate_lcm_const_, rate_common_, std::string("rate_common"));

  parameter_check(cons_abs_lin_x_, cons_speed_l_normal_, std::string("cons_speed_l_normal_mps"));

  const auto gait_toml = locomotion_params_dir + std::string("map_gait.toml");
  if (!gait_interface_.init_gait_map(gait_toml)) {
    return CallbackReturn_T::ERROR;
  }

  callback_group_service =
    this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

  mode_server_ = std::make_unique<
    cyberdog_utils::ActionServer<ChangeMode_T, LifecycleNode_T>>(
    this->shared_from_this(), "checkout_mode",
    std::bind(&MotionManager::checkout_mode, this));

  gait_server_ = std::make_unique<
    cyberdog_utils::ActionServer<ChangeGait_T, LifecycleNode_T>>(
    this->shared_from_this(), "checkout_gait",
    std::bind(&MotionManager::checkout_gait, this));

  monorder_server_ = std::make_unique<
    cyberdog_utils::ActionServer<ExtMonOrder_T, LifecycleNode_T>>(
    this->shared_from_this(), "exe_monorder",
    std::bind(&MotionManager::mon_order_exec, this));

  gait_pub_ = this->create_publisher<Gait_T>(
    "gait_out", rclcpp::SystemDefaultsQoS());

  control_state_pub_ = this->create_publisher<ControlState_T>(
    "status_out", rclcpp::SystemDefaultsQoS());

  odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>(
    "odom_out", rclcpp::SystemDefaultsQoS());

  cmd_pub_ = this->create_publisher<SE3VelocityCMD_T>(
    "cmd_out", rclcpp::SystemDefaultsQoS());

  casual_service_client_ =
    this->create_client<automation_msgs::srv::NavMode>("nav_mode");

  ob_detect_client_ =
    this->create_client<ception_msgs::srv::SensorDetectionNode>("obstacle_detection");

  velocity_sub_ = this->create_subscription<SE3VelocityCMD_T>(
    topic_name_map_["rc_topic"], rclcpp::SystemDefaultsQoS(),
    std::bind(
      &MotionManager::velocity_cmd_callback, this,
      std::placeholders::_1));

  ob_detect_sub_ = this->create_subscription<Around_T>(
    "ObstacleDetection", rclcpp::SystemDefaultsQoS(),
    std::bind(
      &MotionManager::ob_detection_callback, this,
      std::placeholders::_1));

  paras_sub_ = this->create_subscription<Parameters_T>(
    "para_change", rclcpp::SensorDataQoS(),
    std::bind(&MotionManager::paras_callback, this, std::placeholders::_1));

  cau_sub_ = this->create_subscription<NavCaution_T>(
    "nav_status", rclcpp::SystemDefaultsQoS(),
    std::bind(
      &MotionManager::caution_callback, this,
      std::placeholders::_1));

  control_state_sub_ = this->create_subscription<ControlState_T>(
    "status_out", rclcpp::SystemDefaultsQoS(),
    std::bind(
      &MotionManager::control_state_callback, this,
      std::placeholders::_1));

  guard_sub_ = this->create_subscription<Safety_T>(
    "safe_guard", rclcpp::SystemDefaultsQoS(),
    std::bind(&MotionManager::guard_callback, this, std::placeholders::_1));

  tf_broadcaster = std::make_unique<tf2_ros::TransformBroadcaster>(
    this, rclcpp::SystemDefaultsQoS());

  motion_in_ =
    std::make_unique<lcm::LCM>(
    this->get_lcm_url(
      std::string("239.255.76.67"), port_recv_from_motion_,
      ttl_recv_from_motion_));
  motion_out_ =
    std::make_unique<lcm::LCM>(
    this->get_lcm_url(
      std::string("239.255.76.67"), port_send_to_motion_,
      ttl_send_to_motion_));
  state_es_in_ =
    std::make_unique<lcm::LCM>(
    this->get_lcm_url(
      std::string("239.255.76.67"), port_from_odom_,
      ttl_from_odom_));

  motion_in_->subscribe(
    "exec_response", &MotionManager::control_lcm_collection, this);
  state_es_in_->subscribe(
    "state_estimator",
    &MotionManager::statees_lcm_collection, this);

  message_info(get_name() + std::string(" configured"));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T MotionManager::on_activate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" onactivating"));

  cmd_pub_->on_activate();
  control_state_pub_->on_activate();
  gait_pub_->on_activate();
  odom_pub_->on_activate();

  init();

  mode_server_->activate();
  gait_server_->activate();
  monorder_server_->activate();
  // seq_server_->activate();

  message_info(get_name() + std::string(" activated"));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T MotionManager::on_deactivate(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" deactivating"));

  thread_flag_ = false;
  node_exec_.cancel();

  mode_server_->deactivate();
  gait_server_->deactivate();
  monorder_server_->deactivate();
  // seq_server_->deactivate();
  control_state_pub_->on_deactivate();
  gait_pub_->on_deactivate();
  odom_pub_->on_deactivate();
  cmd_pub_->on_deactivate();

  automation_node_thread_->join();
  lcm_control_res_handle_thread_->join();
  lcm_statees_res_handle_thread_->join();
  control_cmd_thread_->join();

  message_info(get_name() + std::string(" deactivated"));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T MotionManager::on_cleanup(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" cleaning up"));

  velocity_sub_.reset();
  ob_detect_sub_.reset();
  guard_sub_.reset();
  gait_pub_.reset();
  control_state_pub_.reset();
  odom_pub_.reset();
  cmd_pub_.reset();

  casual_service_client_.reset();
  ob_detect_client_.reset();

  // Reset service servers
  mode_server_.reset();
  gait_server_.reset();

  // Reset action servers
  monorder_server_.reset();
  // seq_server_.reset();

  // Reset variables
  motion_out_.reset();
  motion_in_.reset();
  state_es_in_.reset();

  automation_manager_node_.reset();
  automation_node_thread_.reset();
  lcm_control_res_handle_thread_.reset();
  lcm_statees_res_handle_thread_.reset();
  control_cmd_thread_.reset();

  message_info(get_name() + std::string(" completely cleaned up"));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T MotionManager::on_shutdown(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" shutting down"));

  message_info(get_name() + std::string(" shut down"));
  return CallbackReturn_T::SUCCESS;
}

CallbackReturn_T MotionManager::on_error(const rclcpp_lifecycle::State &)
{
  message_info(get_name() + std::string(" error raising"));

  message_info(std::string("Subthreads is working , recycling"));
  thread_flag_ = false;
  node_exec_.cancel();

  automation_node_thread_->join();
  lcm_control_res_handle_thread_->join();
  lcm_statees_res_handle_thread_->join();
  control_cmd_thread_->join();

  velocity_sub_.reset();
  ob_detect_sub_.reset();
  gait_pub_.reset();
  control_state_pub_.reset();

  casual_service_client_.reset();
  ob_detect_client_.reset();

  odom_pub_.reset();
  cmd_pub_.reset();

  // Reset service servers
  mode_server_.reset();
  gait_server_.reset();

  // Reset action servers
  monorder_server_.reset();
  // seq_server_.reset();

  // Reset variables
  motion_out_.reset();
  motion_in_.reset();
  state_es_in_.reset();
  automation_node_thread_.reset();
  lcm_control_res_handle_thread_.reset();
  lcm_statees_res_handle_thread_.reset();
  control_cmd_thread_.reset();

  message_info(get_name() + std::string(" error processed"));
  return CallbackReturn_T::SUCCESS;
}

template<typename T>
T MotionManager::limit_data(T in, T min, T max)
{
  if (in > max) {
    in = max;
  }
  if (in < min) {
    in = min;
  }
  return in;
}

void MotionManager::checkout_mode()
{
  auto goal = mode_server_->get_current_goal();
  auto next_mode = goal->modestamped;
  auto feedback = std::make_shared<ModeFB_T>();
  auto result = std::make_unique<ModeRes_T>();

  bool new_request(false);
  bool check_inside(false);
  bool check_submode(false);
  uint8_t inter_changed_(manager::DEFAULT);
  rclcpp::WallRate looprate(rate_common_);
  auto current_time = this->get_clock()->now();
  auto current_mode = robot_control_state_.modestamped;
  auto gait_before = (robot_control_state_.gaitstamped.gait == Gait_T::GAIT_PASSIVE) ?
    Gait_T::GAIT_KNEEL : robot_control_state_.gaitstamped.gait;
  auto async_client_ = rclcpp_action::create_client<ChangeGait_T>(this, "checkout_gait");

  auto mode_str = (mode_label_.find(next_mode.control_mode) == mode_label_.end()) ?
    std::to_string(next_mode.control_mode) :
    mode_label_[next_mode.control_mode];
  auto TAG_MODE = std::string("[Mode_Check]-[") +
    mode_str +
    std::string("] ");

  message_info(
    TAG_MODE +
    std::string("Got mode check request, goal mode is ") +
    mode_label_[next_mode.control_mode]);

  if (!mode_server_ || !mode_server_->is_server_active()) {
    message_info(
      TAG_MODE +
      std::string("Mode Server is not active"));
    result->err_code = ModeRes_T::REJECT;
    mode_server_->succeeded_current(std::move(result));
    return;
  }

  if (current_mode.control_mode >= Mode_T::MODE_MANUAL) {
    reset_velocity();
  }

  #ifndef DEBUG_ALL
  if (check_time_update(
      goal->modestamped.timestamp,
      current_mode.timestamp))
  {
    new_request = true;
  } else {
    message_warn(
      TAG_MODE +
      std::string("Warning, old timestamp, mode ") +
      mode_label_[next_mode.control_mode] +
      std::string(" will not be executed"));
    result->err_code = ModeRes_T::BAD_TIMESTAMP;
  }
  #else
  new_request = true;
  #endif

  if (next_mode.control_mode <= Mode_T::MODE_SEMI) {
    if (next_mode.mode_type != Mode_T::DEFAULT_TYPE) {
      result->err_code = ModeRes_T::UNKNOWN;
      new_request = false;
    }
  } else {
    if (submode_map_.find(
        std::pair<uint8_t, uint8_t>(
          next_mode.control_mode,
          next_mode.mode_type)) == submode_map_.end())
    {
      result->err_code = ModeRes_T::UNKNOWN;
      new_request = false;
    }
  }

  if (automation_manager_node_->get_current_state().id() ==  // NOLINT
    lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE ||
    automation_manager_node_->chainnodes_state_ == manager::PART_DEACTIVE)
  {
    auto current_stamp = this->get_clock()->now();
    automation_manager_node_->auto_check(cyberdog_utils::CHECK_TO_START);  // if part_deactive
    automation_manager_node_->auto_check(cyberdog_utils::CHECK_TO_PAUSE);
    while (automation_manager_node_->chainnodes_state_ != manager::ALL_DEACTIVE &&
      next_mode.control_mode >= Mode_T::MODE_SEMI)
    {
      if (this->get_clock()->now() - current_stamp >= std::chrono::seconds(timeout_manager_)) {
        message_warn(
          TAG_MODE +
          std::string("Subnodes deactivate failed, some nodes running back end"));
        break;
      }
    }
    current_time = this->get_clock()->now();
  }

  if (robot_control_state_.safety.status == Safety_T::LOW_BTR) {
    if (next_mode.control_mode >= Mode_T::MODE_SEMI) {
      result->err_code = ModeRes_T::UNAVAILABLE;
      message_warn(
        TAG_MODE +
        std::string("Battery low, reject high consumption mode"));
      new_request = false;
    }
  }

  while (rclcpp::ok() && new_request) {
    if (mode_server_->is_cancel_requested()) {
      message_info(
        TAG_MODE +
        std::string("Mode Server is canceled"));
      result->err_code = ModeRes_T::CANCELED;
      new_request = false;
      if (next_mode.control_mode >= Mode_T::MODE_SEMI) {
        message_info(
          TAG_MODE +
          std::string("Auto recycle automation nodes"));
        if (automation_manager_node_->get_current_state().id() ==  // NOLINT
          lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE ||
          automation_manager_node_->chainnodes_state_ != manager::ALL_DEACTIVE)
        {
          automation_manager_node_->auto_check(cyberdog_utils::CHECK_TO_PAUSE);
        }
      }
      break;
    }
    if (mode_server_->is_preempt_requested()) {
      auto pending_mode = mode_server_->get_pending_goal().get()->modestamped;
      if (pending_mode.control_mode == next_mode.control_mode &&
        pending_mode.mode_type == next_mode.mode_type)
      {
        message_info(
          TAG_MODE +
          std::string("Pending goal is same with current checking, reject"));
        mode_server_->terminate_pending_goal();
      } else {
        message_info(
          TAG_MODE +
          std::string("New mode checking req received, terminate current now"));
        mode_server_->terminate_current();
        if (gait_server_->is_running()) {
          async_client_->async_cancel_all_goals();
        }
        new_request = false;
        if (next_mode.control_mode >= Mode_T::MODE_SEMI) {
          message_info(
            TAG_MODE +
            std::string("Auto recycle automation nodes"));
          if (automation_manager_node_->get_current_state().id() ==  // NOLINT
            lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE ||
            automation_manager_node_->chainnodes_state_ != manager::ALL_DEACTIVE)
          {
            automation_manager_node_->auto_check(cyberdog_utils::CHECK_TO_PAUSE);
          }
        }
        result->err_code = ModeRes_T::AVOID_PREEMPT;
        break;
      }
    }
    if (new_request) {
      switch (next_mode.control_mode) {
        case Mode_T::MODE_DEFAULT:
        case Mode_T::MODE_MANUAL: {
            bool condition_default = robot_control_state_.gaitstamped.gait != Gait_T::GAIT_KNEEL ||
              robot_control_state_.gaitstamped.gait != Gait_T::GAIT_PASSIVE;
            bool condition_manual = current_mode.control_mode == Mode_T::MODE_DEFAULT ||
              current_mode.control_mode == Mode_T::MODE_LOCK;
            bool condition_pre =
              next_mode.control_mode == Mode_T::MODE_DEFAULT ?
              condition_default : condition_manual;
            Gait_T mode_trig_gait;
            mode_trig_gait.timestamp = this->get_clock()->now();
            mode_trig_gait.gait = (next_mode.control_mode == Mode_T::MODE_DEFAULT) ?
              Gait_T::GAIT_KNEEL :
              Gait_T::GAIT_STAND_R;
            if (condition_pre) {
              if (check_inside) {
                if (this->get_clock()->now() - current_time <=
                  std::chrono::seconds(timeout_gait_ * 2))
                {
                  if (inter_changed_ == manager::SUCCEED) {
                    message_info(
                      TAG_MODE +
                      std::string("Auto ") +
                      gait_label_[mode_trig_gait.gait] +
                      std::string(" succeed"));
                    new_request = false;
                  } else if (inter_changed_ == manager::FAILED) {
                    message_info(
                      TAG_MODE +
                      std::string("Auto ") +
                      gait_label_[mode_trig_gait.gait] +
                      std::string(" failed"));
                    new_request = false;
                    result->err_code = ModeRes_T::FAILED;
                  }
                } else {
                  message_warn(
                    TAG_MODE +
                    std::string("Timeout, canceling mode checking"));
                  result->err_code = ModeRes_T::TIME_OUT;
                  new_request = false;
                }
              } else {
                auto goal = ChangeGait_T::Goal();
                goal.motivation = cyberdog_utils::MODE_TRIG;
                goal.gaitstamped = mode_trig_gait;
                auto goal_options = rclcpp_action::Client<ChangeGait_T>::SendGoalOptions();
                auto gait_result_callback =
                  [&](const GoalHandleGait_T::WrappedResult & result) -> void {
                    switch (result.code) {
                      case rclcpp_action::ResultCode::SUCCEEDED: {
                          if (result.result.get()->succeed) {
                            inter_changed_ = manager::SUCCEED;
                          } else {
                            inter_changed_ = manager::FAILED;
                          }
                          break;
                        }
                      case rclcpp_action::ResultCode::CANCELED:
                      case rclcpp_action::ResultCode::ABORTED: {
                          inter_changed_ = manager::FAILED;
                          break;
                        }
                      default: {
                          break;
                        }
                    }
                  };
                goal_options.result_callback = gait_result_callback;
                async_client_->async_send_goal(goal, goal_options);
                check_inside = true;
              }
              feedback->timestamp = this->get_clock()->now();
              feedback->current_state = ModeFB_T::CHECKING_GAIT;
              mode_server_->publish_feedback(feedback);
            } else {
              new_request = false;
            }
            break;
          }
        case Mode_T::MODE_LOCK: {
            new_request = false;
            break;
          }
        case Mode_T::MODE_SEMI: {
            message_info(
              TAG_MODE +
              std::string("Currently not support mode ") +
              mode_label_[next_mode.control_mode]);
            result->err_code = ModeRes_T::UNKNOWN;
            new_request = false;
            break;
          }
        case Mode_T::MODE_EXPLOR:
        case Mode_T::MODE_TRACK: {
            if (monorder_server_->is_running()) {
              message_warn(
                TAG_MODE +
                std::string("Order is executing, reject other mode checking"));
              result->err_code = ModeRes_T::REJECT;
              new_request = false;
              break;
            }
            if (!check_submode) {
              if (automation_manager_node_->chainnodes_state_ == manager::ALL_ACTIVE) {
                message_info(
                  TAG_MODE +
                  std::string("Subnodes are active now"));
                check_submode = true;
              } else if (automation_manager_node_->get_current_state().id() !=  // NOLINT
                lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE)
              {
                automation_manager_node_->reset_nodes(next_mode);
                automation_manager_node_->auto_check(cyberdog_utils::CHECK_TO_START);
                feedback->timestamp = this->get_clock()->now();
                feedback->current_state = ModeFB_T::WAITING_NODES;
                mode_server_->publish_feedback(feedback);
              } else {
                if (this->get_clock()->now() - current_time >=
                  std::chrono::seconds(timeout_manager_))
                {
                  result->err_code = ModeRes_T::TIME_OUT;
                  result->err_state = feedback->current_state;
                  new_request = false;
                  automation_manager_node_->auto_check(
                    cyberdog_utils::CHECK_TO_PAUSE);
                  break;
                }
                feedback->timestamp = this->get_clock()->now();
                feedback->current_state = ModeFB_T::WAITING_NODES;
                mode_server_->publish_feedback(feedback);
              }
            } else {
              if (check_inside) {
                if (this->get_clock()->now() - current_time <=
                  std::chrono::seconds(timeout_manager_))
                {
                  if (inter_changed_ != manager::DEFAULT) {
                    if (inter_changed_ == manager::SUCCEED) {
                      message_info(
                        TAG_MODE +
                        std::string("Check submode ") +
                        submode_label_[next_mode.mode_type] +
                        std::string(" succeed"));
                      new_request = false;
                    } else if (inter_changed_ == manager::FAILED) {
                      message_info(
                        TAG_MODE +
                        std::string("Check submode ") +
                        submode_label_[next_mode.mode_type] +
                        std::string(" failed"));
                      new_request = false;
                      result->err_code = ModeRes_T::FAILED;
                      result->err_state = feedback->current_state;
                    }
                  }
                } else {
                  message_warn(
                    TAG_MODE +
                    std::string("Timeout, canceling mode checking"));
                  result->err_code = ModeRes_T::TIME_OUT;
                  new_request = false;
                  automation_manager_node_->auto_check(
                    cyberdog_utils::CHECK_TO_PAUSE);
                }
              } else {
                auto request =
                  std::make_unique<automation_msgs::srv::NavMode::Request>();
                request->sub_mode = submode_map_[std::pair<uint8_t, uint8_t>(
                      next_mode.control_mode, next_mode.mode_type)];
                auto client_callback =
                  [&](rclcpp::Client<SubMode_T>::SharedFuture future) {
                    inter_changed_ = manager::DEFAULT;
                    auto result = future.get();
                    if (result->success) {
                      inter_changed_ = manager::SUCCEED;
                    } else {
                      inter_changed_ = manager::FAILED;
                    }
                  };
                casual_service_client_->async_send_request(
                  std::move(request),
                  client_callback);
                message_info(
                  TAG_MODE +
                  std::string("Sent request, waiting for response"));
                check_inside = true;
              }
              feedback->timestamp = this->get_clock()->now();
              feedback->current_state = ModeFB_T::CHECKING_SUBMODE;
              mode_server_->publish_feedback(feedback);
            }
            break;
          }
        default: {
            result->err_code = ModeRes_T::UNKNOWN;
            new_request = false;
            break;
          }
      }
    }
    looprate.sleep();
  }
  if (result->err_code == ModeRes_T::NORMAL) {
    message_info(
      TAG_MODE +
      std::string("Check mode ") +
      mode_label_[next_mode.control_mode] + std::string(", submode ") +
      submode_label_[next_mode.mode_type] + std::string(" succeed"));
    robot_control_state_.modestamped = next_mode;
    publish_control_state(robot_control_state_);
    result->succeed = true;
  } else {
    if (gait_before != robot_control_state_.gaitstamped.gait &&
      result->err_code != ModeRes_T::AVOID_PREEMPT)
    {
      message_warn(
        TAG_MODE +
        std::string("Mode check failed, recovery gait before"));
      Gait_T recover_gait;
      recover_gait.gait = gait_before;
      recover_gait.timestamp = this->get_clock()->now();
      inter_check_gait_sync(recover_gait, timeout_gait_, cyberdog_utils::MODE_TRIG);
    }
    result->succeed = false;
    result->err_state = feedback->current_state;
    message_warn(
      TAG_MODE +
      std::string("Check mode ") +
      mode_label_[next_mode.control_mode] +
      std::string(", submode ") +
      submode_label_[next_mode.mode_type] +
      std::string(" failed. Error code is ") +
      mode_errcode_label_[result->err_code] +
      std::string(", error state is ") +
      mode_errstate_label_[result->err_state]);
  }
  mode_server_->succeeded_current(std::move(result));
}

// Gait checkout action callback
void MotionManager::checkout_gait()
{
  auto goal = gait_server_->get_current_goal();
  auto feedback = std::make_shared<GaitFB_T>();
  auto result = std::make_unique<GaitRes_T>();

  bool new_request(false);
  auto current_motivation = goal->motivation;
  auto goal_gait = goal->gaitstamped;
  auto running_mode = robot_control_state_.modestamped;
  rclcpp::WallRate looprate(rate_common_);
  std::vector<Gait_T> gait_list;

  auto motivation_str =
    (gait_motivation_map_.find(current_motivation) != gait_motivation_map_.end()) ?
    gait_motivation_map_[current_motivation] :
    std::to_string(current_motivation);
  auto gait_str = (gait_label_.find(goal_gait.gait) != gait_label_.end()) ?
    gait_label_[goal_gait.gait] :
    std::to_string(goal_gait.gait);
  auto TAG_GAIT = std::string("[Gait_Check]-[") +
    motivation_str +
    std::string("]-[") +
    gait_str +
    std::string("] ");

  auto goal_gait_t = 9;
  auto current_gait_t = 3;
  std::vector<cyberdog::bridge::GaitMono> gait_to_run_t;

  gait_interface_.get_bridges_list(goal_gait_t, current_gait_t, gait_to_run_t);
  while (gait_to_run_t.size() > 0) {
    message_info(std::string("gait to run is ") + gait_to_run_t.back().get_gait_name());
    gait_to_run_t.pop_back();
  }

  message_info(
    TAG_GAIT +
    std::string("Got gait check request, goal gait is ") +
    gait_label_[goal_gait.gait]);

  if (!gait_server_ || !gait_server_->is_server_active()) {
    message_info(
      TAG_GAIT +
      std::string("Gait Server is not active"));
    result->err_code = GaitRes_T::REJECT;
    gait_server_->succeeded_current(std::move(result));
    return;
  }

  #ifndef DEBUG_ALL
  if (check_time_update(
      goal->gaitstamped.timestamp,
      robot_control_state_.gaitstamped.timestamp))
  {
    new_request = true;
  } else {
    message_warn(
      TAG_GAIT +
      std::string("Warning, old timestamp, gait will not be executed"));
    result->err_code = GaitRes_T::BAD_TIMESTAMP;
  }
  #else
  new_request = true;
  #endif

  if (running_mode.control_mode <= Mode_T::MODE_LOCK &&
    goal_gait.gait != Gait_T::GAIT_PASSIVE &&
    current_motivation != cyberdog_utils::MODE_TRIG)
  {
    message_warn(
      TAG_GAIT +
      std::string("Warning, current mode ") +
      mode_label_[running_mode.control_mode] +
      std::string(
        " do not support checking gait, please check "
        "an available mode before checking gait"));
    result->err_code = GaitRes_T::UNAVAILABLE;
    new_request = false;
  }

  if (gait_motivation_map_.find(current_motivation) == gait_motivation_map_.end()) {
    result->err_code = GaitRes_T::UNAVAILABLE;
    message_warn(
      TAG_GAIT +
      std::string("Unavailable motivation"));
    new_request = false;
  }

  if (robot_control_state_.safety.status == Safety_T::LOW_BTR) {
    if (goal_gait.gait <= Gait_T::GAIT_DEFAULT && goal_gait.gait >= Gait_T::GAIT_BOUND) {
      result->err_code = GaitRes_T::UNAVAILABLE;
      message_warn(
        TAG_GAIT +
        std::string("Battery low, reject dangerous gait"));
      new_request = false;
    }
  }

  if (new_request) {
    Gait_T gait_to_pub;
    gait_list.push_back(goal_gait);
    auto current_gait = (goal_gait.gait >= Gait_T::GAIT_STAND_B) ?
      gait_cached_.gait : robot_control_state_.gaitstamped.gait;
    auto current_time = this->get_clock()->now();

    if (gait_list.back().gait == current_gait) {
      message_info(
        TAG_GAIT +
        std::string("Gait running currently is already ") +
        gait_label_[current_gait]);
      result->succeed = true;
      result->err_code = GaitRes_T::NORMAL;
      gait_list.clear();
    }

    while (rclcpp::ok() && gait_list.size() > 0) {
      if ((robot_control_state_.modestamped != running_mode) &&
        current_motivation != cyberdog_utils::MODE_TRIG)
      {
        message_warn(
          TAG_GAIT +
          std::string("Mode changed, gait checking will be interrupted"));
        result->err_code = GaitRes_T::INTERRUPTED;
        gait_list.clear();
        break;
      }

      if (this->get_clock()->now() - current_time >=
        std::chrono::seconds(timeout_gait_ * 2))
      {
        message_error(
          TAG_GAIT +
          std::string("Stuck for ") +
          std::to_string(timeout_gait_ * 2) +
          std::string(" seconds. Canceling"));
        result->err_gait = feedback->current_checking;
        result->err_code = GaitRes_T::STUCK;
        gait_list.clear();
        break;
      }

      if (gait_server_->is_cancel_requested()) {
        message_info(
          TAG_GAIT +
          std::string("Gait Server is canceled"));
        result->err_code = ModeRes_T::CANCELED;
        gait_list.clear();
        break;
      }

      if (gait_server_->is_preempt_requested()) {
        auto pending_motivation =
          gait_server_->get_pending_goal()->motivation;
        auto pending_gait =
          gait_server_->get_pending_goal()->gaitstamped;
        if (pending_motivation < current_motivation ||
          (pending_motivation == current_motivation && goal_gait != pending_gait))
        {
          message_info(
            TAG_GAIT +
            std::string("High priority preempt request, disable current"));
          result->err_code = GaitRes_T::AVOID_PREEMPT;
          gait_server_->terminate_current();
          gait_list.clear();
          break;
        } else {
          auto motivation_str =
            gait_motivation_map_.find(pending_motivation) != gait_motivation_map_.end() ?
            gait_motivation_map_[pending_motivation] : std::to_string(pending_motivation);
          message_info(
            TAG_GAIT +
            std::string("Low priority preempt [") +
            motivation_str +
            std::string("] request, continue current"));
          gait_server_->terminate_pending_goal();
        }
      }

      current_gait = (goal_gait.gait >= Gait_T::GAIT_STAND_B) ?
        gait_cached_.gait : robot_control_state_.gaitstamped.gait;
      gait_to_pub.gait = Gait_T::GAIT_DEFAULT;
      auto current_sending_ = ros_to_lcm_data_.pattern;

      if (current_gait != Gait_T::GAIT_TRANS || gait_cached_.gait != current_sending_) {
        if (gait_cached_.gait != current_sending_ && current_sending_ != Gait_T::GAIT_TRANS) {
          message_info(
            TAG_GAIT +
            std::string("Gait checking current is not what client request, add [") +
            gait_label_[current_sending_] + std::string("] to resolve conflict"));
          current_gait = current_sending_;
        }
        switch (gait_list.back().gait) {
          case Gait_T::GAIT_PASSIVE: {
              message_info(
                TAG_GAIT +
                std::string("Check ") +
                gait_label_[gait_list.back().gait] +
                std::string(" directly"));
              gait_to_pub.gait = gait_list.back().gait;
              break;
            }
          case Gait_T::GAIT_KNEEL: {
              if (current_gait == Gait_T::GAIT_PASSIVE) {
                message_warn(
                  std::string(
                    TAG_GAIT +
                    std::string("Future condition met, no need to check ") +
                    gait_label_[gait_list.back().gait] +
                    std::string(". Exit")));
                gait_list.clear();
                break;
              } else if (current_gait == Gait_T::GAIT_STAND_R) {
                message_info(
                  std::string(
                    TAG_GAIT +
                    std::string("Condition met, check ") +
                    gait_label_[gait_list.back().gait] +
                    std::string(" directly")));
                gait_to_pub.gait = gait_list.back().gait;
              } else {
                Gait_T pre_gait;
                pre_gait.gait = Gait_T::GAIT_STAND_R;
                message_info(
                  TAG_GAIT +
                  std::string("Condition not met, check to ") +
                  gait_label_[pre_gait.gait]);
                gait_list.push_back(pre_gait);
              }
              break;
            }
          case Gait_T::GAIT_STAND_R: {
              if (current_gait == Gait_T::GAIT_PASSIVE ||
                current_gait == Gait_T::GAIT_KNEEL ||
                current_gait == Gait_T::GAIT_STAND_B)
              {
                message_info(
                  TAG_GAIT +
                  std::string("Condition met, check ") +
                  gait_label_[gait_list.back().gait] +
                  std::string(" directly"));
                gait_to_pub.gait = gait_list.back().gait;
              } else if (current_gait > Gait_T::GAIT_STAND_B &&  // NOLINT
                current_gait < Gait_T::GAIT_DEFAULT)
              {
                Gait_T pre_gait;
                pre_gait.gait = Gait_T::GAIT_STAND_B;
                message_info(
                  TAG_GAIT +
                  std::string("Condition not met, check to ") +
                  gait_label_[pre_gait.gait]);
                gait_list.push_back(pre_gait);
              } else {
                Gait_T pre_gait;
                pre_gait.gait = Gait_T::GAIT_PASSIVE;
                message_info(
                  TAG_GAIT +
                  std::string("Condition not met, check to ") +
                  gait_label_[pre_gait.gait]);
                gait_list.push_back(pre_gait);
              }
              break;
            }
          case Gait_T::GAIT_STAND_B: {
              if (current_gait == Gait_T::GAIT_STAND_R ||
                (current_gait > Gait_T::GAIT_STAND_B && current_gait < Gait_T::GAIT_DEFAULT))
              {
                message_info(
                  TAG_GAIT +
                  std::string("Condition met, check ") +
                  gait_label_[gait_list.back().gait] +
                  std::string(" directly"));
                gait_to_pub.gait = gait_list.back().gait;
              } else {
                Gait_T pre_gait;
                pre_gait.gait = Gait_T::GAIT_STAND_R;
                message_info(
                  TAG_GAIT +
                  std::string("Condition not met, check to ") +
                  gait_label_[pre_gait.gait]);
                gait_list.push_back(pre_gait);
              }
              break;
            }
          case Gait_T::GAIT_WALK:
          case Gait_T::GAIT_SLOW_TROT:
          case Gait_T::GAIT_TROT:
          case Gait_T::GAIT_FLYTROT:
          case Gait_T::GAIT_BOUND:
          case Gait_T::GAIT_PRONK: {
              if (current_gait >= Gait_T::GAIT_STAND_B && current_gait < Gait_T::GAIT_DEFAULT) {
                message_info(
                  TAG_GAIT +
                  std::string("Condition met, check ") +
                  gait_label_[gait_list.back().gait] +
                  std::string(" directly"));
                gait_to_pub.gait = gait_list.back().gait;
              } else {
                Gait_T pre_gait;
                pre_gait.gait = Gait_T::GAIT_STAND_B;
                message_info(
                  TAG_GAIT +
                  std::string("Condition not met, check to ") +
                  gait_label_[pre_gait.gait]);
                gait_list.push_back(pre_gait);
              }
              break;
            }
          default: {
              message_info(
                TAG_GAIT +
                std::string("Unknown gait, reject"));
              result->err_code = GaitRes_T::UNKNOWN;
              gait_list.clear();
              break;
            }
        }  // end of switch
      }
      auto current_time = this->get_clock()->now();
      while (rclcpp::ok() &&
        gait_to_pub.gait != Gait_T::GAIT_DEFAULT)
      {
        if ((robot_control_state_.modestamped.control_mode !=
          running_mode.control_mode ||
          (robot_control_state_.modestamped.control_mode ==
          running_mode.control_mode &&
          robot_control_state_.modestamped.mode_type !=
          running_mode.mode_type)) &&
          current_motivation != cyberdog_utils::MODE_TRIG)
        {
          message_warn(
            TAG_GAIT +
            std::string("Mode changed, gait checking will be canceled"));
          result->err_code = GaitRes_T::CANCELED;
          gait_list.clear();
          break;
        }

        if (gait_server_->is_cancel_requested()) {
          message_info(
            TAG_GAIT +
            std::string("Gait Server is canceled"));
          result->err_code = ModeRes_T::CANCELED;
          break;
        }
        if (gait_server_->is_preempt_requested()) {
          auto pending_motivation =
            gait_server_->get_pending_goal()->motivation;
          auto pending_gait =
            gait_server_->get_pending_goal()->gaitstamped.gait;
          if (pending_motivation < current_motivation ||
            (pending_motivation == current_motivation && goal_gait.gait != pending_gait))
          {
            message_info(
              TAG_GAIT +
              std::string("High priority preempt request, disable current"));
            result->err_code = GaitRes_T::AVOID_PREEMPT;
            gait_server_->terminate_current();
            gait_list.clear();
            break;
          } else {
            auto motivation_str =
              gait_motivation_map_.find(pending_motivation) != gait_motivation_map_.end() ?
              gait_motivation_map_[pending_motivation] : std::to_string(pending_motivation);
            message_info(
              TAG_GAIT +
              std::string("Low priority preempt [") +
              motivation_str +
              std::string("] request, continue current"));
            gait_server_->terminate_pending_goal();
          }
        }

        if (this->get_clock()->now() - current_time >=
          std::chrono::seconds(timeout_gait_))
        {
          message_info(
            TAG_GAIT +
            std::string("Timeout, cancel check ") +
            gait_label_[gait_to_pub.gait]);
          result->err_gait = gait_list.back();
          result->err_code = GaitRes_T::TIME_OUT;
          gait_list.clear();
          break;
        }
        current_gait =
          (gait_to_pub.gait >
          Gait_T::GAIT_STAND_B) ? gait_cached_.gait : robot_control_state_.gaitstamped.gait;

        gait_to_pub.timestamp = this->get_clock()->now();
        publish_gait(gait_to_pub, current_motivation == cyberdog_utils::ORDER_REQ);
        if (current_gait == gait_to_pub.gait && current_gait == gait_cached_.gait) {
          message_info(
            TAG_GAIT +
            std::string("Gait ") +
            gait_label_[current_gait] +
            std::string(" check succeed"));
          gait_list.pop_back();
          break;
        }
        feedback->current_checking = gait_to_pub;
        gait_server_->publish_feedback(feedback);
        looprate.sleep();
      }  // end of int while
    }  // end of ext while
  }  // end of new_request

  if (result->err_code == GaitRes_T::NORMAL) {
    robot_control_state_.gaitstamped.timestamp = goal->gaitstamped.timestamp;
    result->succeed = true;
  } else {
    gait_cached_ = robot_control_state_.gaitstamped;
    result->err_gait = feedback->current_checking;
    result->succeed = false;
    message_warn(
      TAG_GAIT +
      std::string("Check gait execute failed. Error code is ") +
      gait_errcode_label_[result->err_code] +
      std::string(", error gait is ") +
      gait_label_[result->err_gait.gait]);
  }
  gait_server_->succeeded_current(std::move(result));
}

void MotionManager::mon_order_exec()
{
  auto const goal_order = monorder_server_->get_current_goal()->orderstamped;
  auto feedback = std::make_shared<MonOrderFB_T>();
  auto result = std::make_unique<MonOrderRes_T>();

  bool const need_order_list = goal_order.id >= MonOrder_T::MONO_ORDER_HI_FIVE;
  bool interruptible(true);
  bool new_request(false);
  bool ext_interrupt(false);
  bool without_recovery(false);

  auto goal_id = goal_order.id;
  auto running_mode = robot_control_state_.modestamped;
  auto running_gait = gait_cached_;
  auto vcmd_bak_ = ext_velocity_cmd_;

  Gait_T gait_to_use(gait_cached_);
  std::vector<toml::table> order_steps;
  rclcpp::WallRate rate_order(rate_common_);

  auto order_str = (order_label_.count(goal_order.id) == 0) ?
    std::to_string(goal_order.id) :
    order_label_[goal_order.id];
  auto TAG_ORDER = std::string("[Order_Exec]-[") +
    order_str +
    std::string("] ");

  message_info(
    TAG_ORDER +
    std::string("Got order execution request"));

  if (!monorder_server_ || !monorder_server_->is_server_active()) {
    message_info(
      TAG_ORDER +
      std::string("Mon Order Server is not active"));
    result->err_code = MonOrderRes_T::REJECT;
    monorder_server_->succeeded_current(std::move(result));
    return;
  }

  #ifndef DEBUG_ALL
  if (check_time_update(
      goal_order.timestamp,
      robot_control_state_.orderstamped.timestamp))
  {
    new_request = true;
  } else {
    message_info(
      TAG_ORDER +
      std::string("Warning, old timestamp, order ") +
      std::to_string(goal_order.id) +
      std::string(" will not be executed"));
    result->err_code = MonOrderRes_T::BAD_TIMESTAMP;
  }
  #else
  tqdm_single_ = 0;
  new_request = true;
  #endif

  // Mode checking
  if (robot_control_state_.modestamped.control_mode != Mode_T::MODE_MANUAL) {
    message_info(
      TAG_ORDER +
      std::string("Order will not execute in current mode [") +
      mode_label_[robot_control_state_.modestamped.control_mode] + std::string("]"));
    new_request = false;
    result->err_code = MonOrderRes_T::UNAVAILABLE;
  }

  // Battery checking
  if (robot_control_state_.safety.status == Safety_T::LOW_BTR) {
    message_info(
      TAG_ORDER +
      std::string("Battery low, reject all action request"));
    new_request = false;
    result->err_code = MonOrderRes_T::UNAVAILABLE;
  }

  if (new_request) {
    robot_control_state_.orderstamped.id = goal_order.id;
    switch (goal_order.id) {
      // Order without recovery gait before order
      case MonOrder_T::MONO_ORDER_PROSTRATE:
        {
          without_recovery = true;
        }
      // fall through
      case MonOrder_T::MONO_ORDER_STAND_UP:
      case MonOrder_T::MONO_ORDER_STEP_BACK:
      case MonOrder_T::MONO_ORDER_TURN_AROUND:
        {
          if (gait_cached_.gait != order_to_gait_[goal_order.id]) {
            gait_to_use.gait = order_to_gait_[goal_order.id];
            gait_to_use.timestamp = this->get_clock()->now();
            auto check_result = inter_check_gait_sync(
              gait_to_use, timeout_gait_, cyberdog_utils::ORDER_REQ);
            if (!check_result) {
              message_info(
                TAG_ORDER +
                std::string("Gait check failed"));
              new_request = false;
              result->err_code = MonOrderRes_T::FAILED;
            }
          } else {
            message_info(
              TAG_ORDER +
              std::string("Robot is already in current state"));
          }
          break;
        }
      case MonOrder_T::MONO_ORDER_HI_FIVE:
      case MonOrder_T::MONO_ORDER_DANCE:
      case MonOrder_T::MONO_ORDER_SIT:
      case MonOrder_T::MONO_ORDER_SHOW:
        {
          auto order_file = local_params_dir + std::string("/orders/") +
            order_label_[goal_order.id] + std::string(".toml");
          if (!std::experimental::filesystem::exists(order_file)) {
            result->err_code = MonOrderRes_T::FILE_MISSED;
            new_request = false;
          } else {
            auto order_data = toml::parse(order_file);
            order_steps = toml::find<std::vector<toml::table>>(order_data, "step");
            std::reverse(order_steps.begin(), order_steps.end());

            // Check to recovery stand to get ready
            gait_to_use.timestamp = this->get_clock()->now();
            gait_to_use.gait = Gait_T::GAIT_STAND_R;
            if (!inter_check_gait_sync(gait_to_use, timeout_gait_, cyberdog_utils::ORDER_REQ)) {
              result->err_code = MonOrderRes_T::FAILED;
              message_info(
                TAG_ORDER +
                std::string("Recovery stand failed, canceling"));
              new_request = false;
            }
            goal_id = MonOrder_T::MONO_ORDER_LIST;  // temp code
          }
          break;
        }
      case MonOrder_T::MONO_ORDER_WELCOME:
      case MonOrder_T::MONO_ORDER_TURN_OVER:
        {
          // Check to recovery stand to get ready
          gait_to_use.timestamp = this->get_clock()->now();
          gait_to_use.gait = Gait_T::GAIT_STAND_R;
          if (!inter_check_gait_sync(gait_to_use, timeout_gait_, cyberdog_utils::ORDER_REQ)) {
            result->err_code = MonOrderRes_T::FAILED;
            message_info(
              TAG_ORDER +
              std::string("Recovery stand failed, canceling"));
            new_request = false;
          }
          interruptible = false;
          break;
        }
      default:
        {
          message_info(
            TAG_ORDER +
            std::string("Unknown order id [") +
            std::to_string(goal_order.id) +
            std::string("]"));
          new_request = false;
          result->err_code = MonOrderRes_T::UNKNOWN;
          break;
        }
    }

    auto current_time = this->get_clock()->now();
    auto cons_speed = goal_order.id == MonOrder_T::MONO_ORDER_TURN_AROUND ?
      cons_speed_a_normal_ : cons_speed_l_normal_;
    auto running_time = static_cast<int64_t>(std::abs(std::ceil(goal_order.para / cons_speed)));
    // running cycle
    while (rclcpp::ok() && new_request) {
      if (this->get_clock()->now() - current_time >=
        std::chrono::seconds(timeout_order_ / 4))
      {
        message_warn(
          TAG_ORDER +
          std::string("Stuck for ") +
          std::to_string(timeout_order_ / 4) +
          std::string(" seconds. Canceling"));
        result->err_code = MonOrderRes_T::STUCK;
        new_request = false;
        break;
      }

      if (robot_control_state_.modestamped != running_mode) {
        message_warn(
          TAG_ORDER +
          std::string("Mode changed, order executing will be canceled"));
        result->err_code = MonOrderRes_T::CANCELED;
        new_request = false;
        if (robot_control_state_.modestamped.control_mode < Mode_T::MODE_MANUAL) {
          ext_interrupt = true;
        }
        break;
      }

      if ((gait_cached_.gait != gait_to_use.gait) && interruptible) {
        message_warn(
          TAG_ORDER +
          std::string("Gait interruption occured, order executing will be canceled"));
        result->err_code = MonOrderRes_T::INTERRUPTED;
        ext_interrupt = true;
        new_request = false;
        break;
      }

      if ((vcmd_bak_ != ext_velocity_cmd_) && interruptible) {
        message_warn(
          TAG_ORDER +
          std::string("VCMD interruption occured, order executing will be canceled"));
        result->err_code = MonOrderRes_T::INTERRUPTED;
        new_request = false;
        break;
      }

      if (monorder_server_->is_cancel_requested()) {
        message_info(
          TAG_ORDER +
          std::string("Order request is canceled"));
        result->err_code = ModeRes_T::CANCELED;
        new_request = false;
        break;
      }

      if (monorder_server_->is_preempt_requested()) {
        message_info(
          TAG_ORDER +
          std::string("A preempt order received"));
        if (interruptible) {
          monorder_server_->terminate_current();
          result->err_code = MonOrderRes_T::INTERRUPTED;
          new_request = false;
          message_info(
            TAG_ORDER +
            std::string("Order could be interrupted, terminate current order"));
          break;
        } else {
          monorder_server_->terminate_pending_goal();
          message_info(
            TAG_ORDER +
            std::string("Order could NOT be interrupted, terminate pending order"));
        }
      }

      if (!need_order_list) {  // gait-like order
        auto cost_time = this->get_clock()->now() - current_time;

        if (goal_order.para == 0.0) {
          break;
        } else {  // movement order
          if (cost_time > std::chrono::seconds(running_time)) {
            message_info(
              TAG_ORDER +
              std::string("Movement order is complete"));
            new_request = false;
            break;
          } else {  // remain_time < running_time
            auto velocity_order = std::make_unique<SE3VelocityCMD_T>();
            velocity_order->sourceid = SE3VelocityCMD_T::INTERNAL;
            velocity_order->velocity.frameid.id = FrameID_T::BODY_FRAME;
            velocity_order->velocity.timestamp = this->get_clock()->now();
            velocity_order->velocity.linear_x =
              (goal_order.id == MonOrder_T::MONO_ORDER_STEP_BACK) ?
              -cons_speed_l_normal_ : 0;
            velocity_order->velocity.angular_z =
              (goal_order.id == MonOrder_T::MONO_ORDER_TURN_AROUND) ?
              cons_speed_a_normal_ : 0;

            publish_velocity(std::move(velocity_order));
            feedback->current_pose = robot_control_state_.posestamped;
            feedback->order_executing = goal_order;
            auto cost_time_mili = cost_time.to_chrono<std::chrono::milliseconds>().count();
            feedback->process_rate = std::floor(cost_time_mili / running_time * 100);
            monorder_server_->publish_feedback(feedback);
          }
        }
      } else {  // list order
        auto condition_timeout = this->get_clock()->now() -
          current_time >= std::chrono::seconds(timeout_order_);
        if (order_running_ != goal_id) {
          reset_velocity(goal_id);
        } else if (condition_timeout) {
          result->err_code = MonOrderRes_T::TIME_OUT;
          new_request = false;
          break;
        }
        if (order_steps.size() > 0) {
          auto traj_out_ = std::make_unique<trajectory_command_lcmt>();
          toml_to_lcm(order_steps.back(), *traj_out_);
          motion_out_->publish("motion-list", &(*traj_out_));
          order_steps.pop_back();
        } else {
          if (tqdm_single_ >= 100) {
            break;
          } else {
            feedback->process_rate = tqdm_single_;
            feedback->current_pose = robot_control_state_.posestamped;
            feedback->order_executing = goal_order;
            monorder_server_->publish_feedback(feedback);
          }
        }
      }
      if (order_steps.size() == 0) {
        rate_order.sleep();
      }
    }
  }

  // recovery stand after internal order execution
  if (need_order_list && !ext_interrupt && result->err_code != MonOrderRes_T::UNAVAILABLE) {
    reset_velocity(MonOrder_T::MONO_ORDER_NULL);
    gait_to_use.gait = Gait_T::GAIT_STAND_R;
    gait_to_use.timestamp = this->get_clock()->now();
    if (!inter_check_gait_sync(gait_to_use, timeout_gait_, cyberdog_utils::ORDER_REQ)) {
      result->err_code = MonOrderRes_T::RECOV_FAILED;
    }
  }

  // recovery gait before order execution
  if (running_gait.gait >= Gait_T::GAIT_STAND_B && !without_recovery && !ext_interrupt &&
    result->err_code != MonOrderRes_T::UNAVAILABLE)
  {
    gait_to_use.gait = running_gait.gait;
    gait_to_use.timestamp = this->get_clock()->now();
    if (!inter_check_gait_sync(gait_to_use, timeout_gait_, cyberdog_utils::GAIT_TRIG)) {
      result->err_code = MonOrderRes_T::RECOV_FAILED;
    }
  }

  if (result->err_code == MonOrderRes_T::NORMAL) {
    message_info(
      TAG_ORDER +
      std::string("Order execution is succeed"));
    robot_control_state_.orderstamped.timestamp = this->get_clock()->now();
    result->succeed = true;
  } else {
    message_info(
      TAG_ORDER +
      std::string("Order execution is failed, error code is ") +
      order_errorcode_label_[result->err_code]);

    result->succeed = false;
  }

  robot_control_state_.orderstamped.id = MonOrder_T::MONO_ORDER_NULL;
  monorder_server_->succeeded_current(std::move(result));
}

void MotionManager::velocity_cmd_callback(
  const SE3VelocityCMD_T::SharedPtr msg)
{
  auto current_mode = robot_control_state_.modestamped;
  auto condition_internal = msg->sourceid == SE3VelocityCMD_T::INTERNAL;
  auto condition_remotec = msg->sourceid == SE3VelocityCMD_T::REMOTEC;
  auto condition_navigator = msg->sourceid == SE3VelocityCMD_T::NAVIGATOR;
  auto TAG_VCMD = std::string("[VCMD_Update] ");
  switch (current_mode.control_mode) {
    case Mode_T::MODE_MANUAL:
      {
        if (condition_remotec || condition_internal) {
          publish_velocity(msg);
          ext_velocity_cmd_ = *msg;
        }
        break;
      }
    case Mode_T::MODE_EXPLOR:
      {
        if (condition_remotec | condition_navigator | condition_internal) {
          publish_velocity(msg);
          ext_velocity_cmd_ = *msg;
        }
        break;
      }
    case Mode_T::MODE_TRACK:
      {
        if (condition_navigator | condition_internal) {
          publish_velocity(msg);
          ext_velocity_cmd_ = *msg;
        }
        break;
      }
    default:
      {
        message_warn(
          TAG_VCMD +
          std::string("Not support velocity command right now"));
        break;
      }
  }
}

void MotionManager::ob_detection_callback(const Around_T::SharedPtr msg)
{
  obstacle_data_ = *msg;
}

void MotionManager::paras_callback(const Parameters_T::SharedPtr msg)
{
  auto TAG_PARAM = std::string("[Param_Update] ");
  #ifndef DEBUG_ALL
  if (!check_time_update(msg->timestamp, robot_control_state_.parastamped.timestamp)) {
    message_error(
      TAG_PARAM +
      std::string("Timestamp is old. Return"));
    return;
  }
  #endif

  if (msg->body_height == 0 || msg->gait_height == 0) {
    message_error(
      TAG_PARAM +
      std::string("Got zero in paras! Return"));
    return;
  } else {
    message_info(
      TAG_PARAM +
      std::string("Body height is ") +
      std::to_string(msg->body_height) +
      std::string(", gait height is ") +
      std::to_string(msg->gait_height));
    robot_control_state_.parastamped = *msg;
    publish_paras(robot_control_state_.parastamped);
  }
}

void MotionManager::caution_callback(const NavCaution_T::SharedPtr msg)
{
  nav_caution_ = *msg;
}

void MotionManager::control_state_callback(const ControlState_T::SharedPtr msg)
{
  auto TAG_STATE = std::string("[State_Detection] ");
  if (check_motor_errflag() ||  // passive when motor error
    (msg->modestamped.control_mode == Mode_T::MODE_LOCK &&  // Lock Detection
    msg->gaitstamped.gait != Gait_T::GAIT_PASSIVE &&
    !gait_server_->is_running() && !mode_server_->is_running()))  // avoid mode trigger gait
  {
    Gait_T passive_gait;
    passive_gait.gait = Gait_T::GAIT_PASSIVE;
    passive_gait.timestamp = this->get_clock()->now();
    message_info(
      TAG_STATE +
      std::string("Locking state & Error state detected. Checking to passive"));
    publish_gait(passive_gait);
  }
}

void MotionManager::guard_callback(const Safety_T::SharedPtr msg)
{
  auto TAG_GUARD = std::string("[Guard_Detection] ");

  if (robot_control_state_.safety != *msg) {
    if (msg->status == Safety_T::LOW_BTR) {
      cons_abs_lin_x_ *= scale_low_btr_;
      cons_abs_lin_y_ *= scale_low_btr_;
      cons_abs_ang_p_ *= scale_low_btr_;
      cons_abs_ang_r_ *= scale_low_btr_;
      cons_abs_ang_y_ *= scale_low_btr_;
      message_info(
        TAG_GUARD +
        std::string("Battery low, reset max velocity"));
    }
    if (msg->status == Safety_T::NORMAL) {
      cons_abs_lin_x_ /= scale_low_btr_;
      cons_abs_lin_y_ /= scale_low_btr_;
      cons_abs_ang_p_ /= scale_low_btr_;
      cons_abs_ang_r_ /= scale_low_btr_;
      cons_abs_ang_y_ /= scale_low_btr_;
      message_info(
        TAG_GUARD +
        std::string("Battery recovery, reset max velocity"));
    }
  }

  robot_control_state_.safety = *msg;

  auto condition_timeout = this->get_clock()->now() - last_motion_time_ >=
    std::chrono::milliseconds(timeout_motion_);
  auto condition_mode_valid = robot_control_state_.modestamped.control_mode >= Mode_T::MODE_MANUAL;
  auto condition_running = robot_control_state_.gaitstamped.gait > Gait_T::GAIT_STAND_B;
  auto condition_order_pause = !monorder_server_->is_running();
  if (condition_timeout && condition_mode_valid && condition_running && condition_order_pause) {
    reset_velocity();
  }
}

void MotionManager::control_lcm_collection(
  const lcm::ReceiveBuffer * rbuf, const std::string & channel,
  const motion_control_response_lcmt * lcm_data)
{
  (void)rbuf;
  (void)channel;

  robot_control_state_.timestamp = this->get_clock()->now();

  robot_control_state_.error_flag.exist_error = lcm_data->error_flag.exist_error;
  robot_control_state_.error_flag.footpos_error = lcm_data->error_flag.footpos_error;
  std::copy_n(
    std::begin(lcm_data->error_flag.motor_error),
    sizeof(lcm_data->error_flag.motor_error) / sizeof(lcm_data->error_flag.motor_error[0]),
    std::begin(robot_control_state_.error_flag.motor_error));
  robot_control_state_.error_flag.ori_error = lcm_data->error_flag.ori_error;
  robot_control_state_.foot_contact = lcm_data->foot_contact;
  robot_control_state_.gaitstamped.gait = lcm_data->pattern;
  robot_control_state_.cached_gait = gait_cached_;

  order_running_ = lcm_data->order;
  tqdm_single_ = lcm_data->order_process_bar;

  if (response_count_ >= std::ceil(rate_lcm_const_ / rate_output_) - 1 &&
    rclcpp::ok() && thread_flag_)
  {
    publish_control_state(robot_control_state_);
    response_count_ = 0;
  }
}

void MotionManager::statees_lcm_collection(
  const lcm::ReceiveBuffer * rbuf, const std::string & channel,
  const state_estimator_lcmt * lcm_data)
{
  (void)rbuf;
  (void)channel;
  auto lcm_time = std::chrono::nanoseconds(lcm_data->timestamp);
  Time_T odom_stamp;
  odom_stamp.sec =
    std::chrono::duration_cast<std::chrono::seconds>(lcm_time).count();
  odom_stamp.nanosec = (lcm_time -
    std::chrono::duration_cast<std::chrono::seconds>(lcm_time)).count();

  robot_control_state_.velocitystamped.timestamp = odom_stamp;
  robot_control_state_.velocitystamped.linear_x = lcm_data->vBody[0];
  robot_control_state_.velocitystamped.linear_y = lcm_data->vBody[1];
  robot_control_state_.velocitystamped.linear_z = lcm_data->vBody[2];
  robot_control_state_.velocitystamped.angular_x = lcm_data->omegaBody[0];
  robot_control_state_.velocitystamped.angular_y = lcm_data->omegaBody[1];
  robot_control_state_.velocitystamped.angular_z = lcm_data->omegaBody[2];

  robot_control_state_.posestamped.timestamp = odom_stamp;
  robot_control_state_.posestamped.position_x = lcm_data->p[0];
  robot_control_state_.posestamped.position_y = lcm_data->p[1];
  robot_control_state_.posestamped.position_z = lcm_data->p[2];
  robot_control_state_.posestamped.rotation_w = lcm_data->quat[0];
  robot_control_state_.posestamped.rotation_x = lcm_data->quat[1];
  robot_control_state_.posestamped.rotation_y = lcm_data->quat[2];
  robot_control_state_.posestamped.rotation_z = lcm_data->quat[3];

  if (odom_count_ >= std::ceil(rate_lcm_const_ / rate_odom_) - 1 &&
    rclcpp::ok() && thread_flag_)
  {
    robot_body_tf_.header.stamp = odom_stamp;
    robot_body_tf_.transform.rotation.w = lcm_data->quat[0];
    robot_body_tf_.transform.rotation.x = lcm_data->quat[1];
    robot_body_tf_.transform.rotation.y = lcm_data->quat[2];
    robot_body_tf_.transform.rotation.z = lcm_data->quat[3];
    robot_body_tf_.transform.translation.x = lcm_data->p[0];
    robot_body_tf_.transform.translation.y = lcm_data->p[1];
    robot_body_tf_.transform.translation.z = lcm_data->p[2];

    odom_.header.stamp = odom_stamp;
    odom_.twist.twist.linear.x = lcm_data->vBody[0];
    odom_.twist.twist.linear.y = lcm_data->vBody[1];
    odom_.twist.twist.linear.z = lcm_data->vBody[2];
    odom_.twist.twist.angular.x = lcm_data->omegaBody[0];
    odom_.twist.twist.angular.y = lcm_data->omegaBody[1];
    odom_.twist.twist.angular.z = lcm_data->omegaBody[2];
    odom_.pose.pose.position.x = lcm_data->p[0];
    odom_.pose.pose.position.y = lcm_data->p[1];
    odom_.pose.pose.position.z = lcm_data->p[2];
    odom_.pose.pose.orientation.w = lcm_data->quat[0];
    odom_.pose.pose.orientation.x = lcm_data->quat[1];
    odom_.pose.pose.orientation.y = lcm_data->quat[2];
    odom_.pose.pose.orientation.z = lcm_data->quat[3];
    tf_broadcaster->sendTransform(robot_body_tf_);
    publish_odom(odom_);
    odom_count_ = 0;
  }
}

void MotionManager::recv_lcm_control_handle()
{
  rclcpp::WallRate r_wait(rate_wait_loop_);

  while (rclcpp::ok() && thread_flag_) {
    while (motion_in_->handleTimeout(timeout_lcm_) > 0 && rclcpp::ok() && thread_flag_) {
      response_count_++;
    }
    #ifndef DEBUG_ALL
    message_info(std::string("[Offline] Motion state response is offline. Clear Velocity & Pose"));
    response_count_ = 0;
    auto velocity_zero = SE3Velocity_T();
    auto pose_zero = SE3Pose_T();
    velocity_zero.timestamp = this->get_clock()->now();
    velocity_zero.frameid.id = FrameID_T::ODOM_FRAME;
    pose_zero.timestamp = this->get_clock()->now();
    pose_zero.frameid.id = FrameID_T::ODOM_FRAME;
    robot_control_state_.set__velocitystamped(velocity_zero);
    robot_control_state_.set__posestamped(pose_zero);
    #endif
    r_wait.sleep();
  }
}

void MotionManager::recv_lcm_statees_handle()
{
  rclcpp::WallRate r_wait(rate_wait_loop_);

  while (rclcpp::ok() && thread_flag_) {
    while (state_es_in_->handleTimeout(timeout_lcm_) > 0 &&
      (robot_control_state_.modestamped.control_mode >= Mode_T::MODE_MANUAL ||
      mode_server_->is_running()) &&
      rclcpp::ok() && thread_flag_)
    {
      odom_count_++;
    }
    #ifndef DEBUG_ALL
    message_info(std::string("[Offline] Odom state response is offline. Clear Velocity & Pose"));
    odom_count_ = 0;
    auto twist_zero = geometry_msgs::msg::TwistWithCovariance();
    auto pose_zero = geometry_msgs::msg::PoseWithCovariance();
    odom_.header.set__stamp(this->get_clock()->now());
    odom_.set__twist(twist_zero);
    odom_.set__pose(pose_zero);
    #endif
    r_wait.sleep();
  }
}

void MotionManager::publish_velocity(
  const SE3VelocityCMD_T::SharedPtr velocity_out,
  const int8_t ORDER_TYPE)
{
  auto TAG_VCMD = std::string("[VCMD_Publish]");
  if (source_id_map_.find(velocity_out->sourceid) == source_id_map_.end()) {
    message_warn(
      TAG_VCMD +
      std::string("Source id [") +
      std::to_string(velocity_out->sourceid) +
      std::string("] unknown. Ignore it"));
    return;
  }

  #ifndef DEBUG_ALL
  if (check_time_update(velocity_out->velocity.timestamp, last_motion_time_)) {
    last_motion_time_ = velocity_out->velocity.timestamp;
  } else {
    message_warn(
      TAG_VCMD +
      std::string("TimeStamp is invalid. Ignore it"));
    return;
  }
  #else
  last_motion_time_ = this->get_clock()->now();
  #endif

  if (robot_control_state_.modestamped.control_mode < Mode_T::MODE_MANUAL) {
    message_warn(
      TAG_VCMD +
      std::string("Robot ignores all velocity command in current mode : ") +
      mode_label_[robot_control_state_.modestamped.control_mode]);
    return;
  }
  if (velocity_out->velocity.frameid.id != FrameID_T::BODY_FRAME) {
    message_warn(
      TAG_VCMD +
      std::string("Frame ID is not [BODY_FRAME], Ignore it"));
    return;
  }

  if (cmd_pub_->is_activated() &&
    this->count_subscribers(cmd_pub_->get_topic_name()) > 0)
  {
    cmd_pub_->publish(*velocity_out);
  }

  ros_to_lcm_data_.linear[0] =
    limit_data(
    velocity_out->velocity.linear_x, 0 - cons_abs_lin_x_,
    cons_abs_lin_x_);
  ros_to_lcm_data_.linear[1] =
    limit_data(
    velocity_out->velocity.linear_y, 0 - cons_abs_lin_y_,
    cons_abs_lin_y_);
  ros_to_lcm_data_.linear[2] =
    limit_data(
    velocity_out->velocity.linear_z, 0 - cons_abs_lin_x_,
    cons_abs_lin_x_);

  ros_to_lcm_data_.angular[0] =
    limit_data(
    velocity_out->velocity.angular_x, 0 - cons_abs_ang_r_,
    cons_abs_ang_r_);
  ros_to_lcm_data_.angular[1] =
    limit_data(
    velocity_out->velocity.angular_y, 0 - cons_abs_ang_p_,
    cons_abs_ang_p_);
  ros_to_lcm_data_.angular[2] =
    limit_data(
    velocity_out->velocity.angular_z, 0 - cons_abs_ang_y_,
    cons_abs_ang_y_);

  ros_to_lcm_data_.order = ORDER_TYPE;
}

void MotionManager::publish_gait(const Gait_T & gait_out, const bool order_req)
{
  ros_to_lcm_data_.pattern = gait_out.gait;
  gait_cached_ = gait_out;

  ros_to_lcm_data_.angular[0] = 0;
  ros_to_lcm_data_.angular[1] = 0;
  ros_to_lcm_data_.angular[2] = 0;

  auto condition_mode_avai = robot_control_state_.modestamped.control_mode >= Mode_T::MODE_MANUAL;
  if (condition_mode_avai && !order_req) {
    reset_velocity();
  }

  if (gait_out.gait < Gait_T::GAIT_STAND_B) {
    gait_pub_->publish(gait_out);
  }
}

void MotionManager::publish_control_state(const ControlState_T & control_state_out)
{
  if (control_state_pub_->is_activated() &&
    this->count_subscribers(control_state_pub_->get_topic_name()) > 0)
  {
    control_state_pub_->publish(control_state_out);
  }
}

void MotionManager::publish_paras(const Parameters_T paras_out)
{
  ros_to_lcm_data_.body_height = paras_out.body_height;
  ros_to_lcm_data_.gait_height = paras_out.gait_height;
}

void MotionManager::init()
{
  // Variable initialization
  thread_flag_ = true;
  tqdm_single_ = 0;
  order_running_ = MonOrder_T::MONO_ORDER_NULL;
  response_count_ = 0;
  odom_count_ = 0;
  last_motion_time_ = this->get_clock()->now();

  robot_control_state_ = ControlState_T();
  robot_control_state_.orderstamped.timestamp = this->get_clock()->now();
  robot_control_state_.modestamped.timestamp = this->get_clock()->now();
  robot_control_state_.gaitstamped.timestamp = this->get_clock()->now();
  robot_control_state_.gaitstamped.gait = Gait_T::GAIT_DEFAULT;
  robot_control_state_.velocitystamped.frameid.id = FrameID_T::ODOM_FRAME;
  robot_control_state_.posestamped.frameid.id = FrameID_T::ODOM_FRAME;
  robot_body_tf_.header.frame_id = std::string("odom");
  robot_body_tf_.child_frame_id = "base_footprint";
  odom_ = nav_msgs::msg::Odometry();
  odom_.header.frame_id = std::string("odom");
  odom_.child_frame_id = std::string("base_footprint");

  automation_manager_node_ =
    std::make_shared<manager::AutomationManager>("multi");
  automation_node_thread_ =
    std::make_unique<std::thread>(&MotionManager::automation_node_spin, this);
  lcm_control_res_handle_thread_ = std::make_unique<std::thread>(
    &MotionManager::recv_lcm_control_handle, this);
  lcm_statees_res_handle_thread_ = std::make_unique<std::thread>(
    &MotionManager::recv_lcm_statees_handle, this);
  control_cmd_thread_ = std::make_unique<std::thread>(
    &MotionManager::control_cmd_spin, this);

  // Parameter initialization
  Parameters_T default_para;
  default_para.timestamp = this->get_clock()->now();
  default_para.body_height = cons_default_body_;
  default_para.gait_height = cons_default_gait_;
  publish_paras(default_para);

  // Gait initialization
  Gait_T gait_init;
  gait_init.gait = Gait_T::GAIT_PASSIVE;
  gait_init.timestamp = this->get_clock()->now();
  publish_gait(gait_init);
  auto goal_gait_t = 9;
  auto current_gait_t = 3;
  std::vector<cyberdog::bridge::GaitMono> gait_to_run_t;

  auto time_a = get_clock()->now();
  gait_interface_.get_bridges_list(goal_gait_t, current_gait_t, gait_to_run_t);
  while (gait_to_run_t.size() > 0) {
    message_info(std::string("gait to run is ") + gait_to_run_t.back().get_gait_name());
    gait_to_run_t.pop_back();
  }
  auto time_b = get_clock()->now();
  auto time_diff = time_b - time_a;
  message_info(std::string("time diff is ") + std::to_string(time_diff.seconds()));
}

std_msgs::msg::Header MotionManager::return_custom_header(std::string frame_id)
{
  std_msgs::msg::Header msg;
  msg.frame_id = frame_id;
  msg.stamp = this->get_clock()->now();

  return msg;
}

bool MotionManager::check_time_update(
  Time_T coming_time,
  Time_T stash_time)
{
  if (coming_time.sec > stash_time.sec ||
    (coming_time.sec == stash_time.sec &&
    coming_time.nanosec >= stash_time.nanosec))
  {
    return true;
  } else {
    return false;
  }
}

bool MotionManager::inter_check_gait_sync(
  const Gait_T & goal_gait,
  const uint32_t & timeout,
  const uint8_t & priority)
{
  auto goal = ChangeGait_T::Goal();
  bool rtn_(false);
  auto sync_client_ = rclcpp_action::create_client<ChangeGait_T>(this, "checkout_gait");

  goal.motivation = priority;
  goal.gaitstamped = goal_gait;

  auto goal_handle = sync_client_->async_send_goal(goal);

  auto result = sync_client_->async_get_result(goal_handle.get());
  result.wait_for(std::chrono::seconds(timeout));
  if (goal_handle.get()->is_result_aware()) {
    if (result.get().result->succeed) {
      rtn_ = true;
    }
  }

  return rtn_;
}

void MotionManager::publish_odom(const nav_msgs::msg::Odometry & odom_out)
{
  if (odom_pub_->is_activated() &&
    this->count_subscribers(odom_pub_->get_topic_name()) > 0)
  {
    odom_pub_->publish(odom_out);
  }
}

void MotionManager::automation_node_spin()
{
  node_exec_.add_node(automation_manager_node_->get_node_base_interface());
  node_exec_.spin();
  node_exec_.remove_node(automation_manager_node_->get_node_base_interface());
}

void MotionManager::control_cmd_spin()
{
  // Send LCM message
  rclcpp::WallRate control_rate_(rate_control_);
  bool is_zeros;
  auto current_gait = robot_control_state_.gaitstamped;
  auto current_motion_time = last_motion_time_;
  auto TAG_MOVEMENT = std::string("[Movement_Detection] ");
  Gait_T gait_out;

  while (rclcpp::ok() && thread_flag_) {
    /*
    Movement detection. QP stand when velocity command is zeros.
  */
    current_gait = robot_control_state_.gaitstamped;
    auto cmd_gait = ros_to_lcm_data_.pattern;
    if (robot_control_state_.modestamped.control_mode >= Mode_T::MODE_MANUAL &&
      current_motion_time != last_motion_time_)
    {
      // Check output value is zero
      is_zeros = true;
      for (const auto & value : ros_to_lcm_data_.linear) {
        is_zeros &= (abs(value) <= 1e-4);
      }
      for (const auto & value : ros_to_lcm_data_.angular) {
        is_zeros &= (abs(value) <= 1e-4);
      }
      if (is_zeros &&
        (current_gait.gait > Gait_T::GAIT_STAND_B || cmd_gait > Gait_T::GAIT_STAND_B))
      {
        ros_to_lcm_data_.pattern = Gait_T::GAIT_STAND_B;
        gait_out.timestamp = this->get_clock()->now();
        gait_out.gait = ros_to_lcm_data_.pattern;
        gait_pub_->publish(gait_out);
        message_info(
          TAG_MOVEMENT +
          std::string("Stop when zero"));
        for (auto & value : ros_to_lcm_data_.linear) {
          value = static_cast<double>(0.0);
        }
        for (auto & value : ros_to_lcm_data_.angular) {
          value = static_cast<double>(0.0);
        }
        message_info(
          TAG_MOVEMENT +
          std::string("Clear velocity cache"));
      } else if (!is_zeros) {
        message_info(
          TAG_MOVEMENT +
          std::string("Run with cached gait ") +
          gait_label_[gait_cached_.gait]);
        ros_to_lcm_data_.pattern = gait_cached_.gait;
        gait_out.timestamp = this->get_clock()->now();
        gait_out.gait = ros_to_lcm_data_.pattern;
        gait_pub_->publish(gait_out);
      }
      current_motion_time = last_motion_time_;
    }

    #ifdef DEBUG_ALL
    #ifdef DEBUG_MOCK
    // [Mock] Set gait directly
    if (ros_to_lcm_data_.pattern != Gait_T::GAIT_TRANS) {
      robot_control_state_.gaitstamped.gait = ros_to_lcm_data_.pattern;
    }
    robot_control_state_.orderstamped.id = ros_to_lcm_data_.order;

    static uint8_t cnt_(0);
    if (robot_control_state_.orderstamped.id != MonOrder_T::MONO_ORDER_NULL) {
      if (tqdm_single_ == 100) {
        tqdm_single_ = 0;
      } else {
        if (cnt_ >= 1) {
          cnt_ = 0;
          tqdm_single_++;
        } else {
          cnt_++;
        }
      }
    }
    #endif
    #ifdef DEBUG_MOTION
    // [Log] Output current gait
    if (ros_to_lcm_data_.pattern > Gait_T::GAIT_TRANS) {
      message_info(std::string("Current gait is ") + gait_label_[ros_to_lcm_data_.pattern]);
    }
    #endif
    #endif

    motion_out_->publish("exec_request", &ros_to_lcm_data_);

    control_rate_.sleep();
  }
}

template<typename T>
void MotionManager::parameter_check(
  const T & extreme_value, T & value_to_check,
  const std::string & value_name, const bool & max_check)
{
  auto condition_extreme =
    max_check ? (value_to_check > extreme_value) : (value_to_check < extreme_value);
  auto string_extreme = max_check ? std::string("is bigger than extreme value ") : std::string(
    "is smaller than extreme value ");
  if (condition_extreme) {
    message_warn(
      std::string("Value ") +
      value_name + std::string(" invalid. Reset it"));
    value_to_check = extreme_value;
  }
}

void MotionManager::reset_velocity(const uint8_t & order_id)
{
  auto velocity_zero = std::make_unique<SE3VelocityCMD_T>();
  velocity_zero->sourceid = SE3VelocityCMD_T::INTERNAL;
  velocity_zero->velocity.frameid.id = FrameID_T::BODY_FRAME;
  velocity_zero->velocity.timestamp = this->get_clock()->now();
  publish_velocity(std::move(velocity_zero), order_id);
}

bool MotionManager::check_motor_errflag(bool show_all)
{
  const char leg_pos[] = {'x', 'y', 'z'};
  static ErrorFlag_T error_flag;
  bool isError =
    (robot_control_state_.error_flag.exist_error != 0 ||
    robot_control_state_.error_flag.ori_error != 0);
  if (error_flag != robot_control_state_.error_flag || show_all) {
    error_flag = robot_control_state_.error_flag;
    std::string err_T = "\n\t|--------------------Motor-State--------------------";
    err_T += "\n\t|\t\t Exist_Error : ";
    error_flag.exist_error ? err_T += "error " : err_T += "normal";
    err_T += "\n\t|\t\tPosture_State: ";
    error_flag.ori_error ? err_T += "error " : err_T += "normal";
    if (isError || show_all) {
      err_T += "\n\t|---------------------Detailes:---------------------\n\t|Footpos_state:";
      int32_t footpos_error = error_flag.footpos_error;
      for (int a = 0; a < 4; a++) {
        err_T += "\n\t|\tLeg[" + std::to_string(a) + "]  ";
        for (int x = 0; x < 3; x++) {
          err_T += leg_pos[x];
          footpos_error & 0x1 ? err_T += ":Error    " : err_T += ":Normal   ";
          footpos_error >>= 1;
        }
        footpos_error >>= 5;
      }
      int motor_num = sizeof(error_flag.motor_error) / sizeof(error_flag.motor_error[0]);
      for (int a = 0; a < motor_num; a++) {
        int32_t motor = error_flag.motor_error[a];
        err_T += "\n\t|Motor[" + std::to_string(a) + "]:";
        if (motor == 0) {err_T += " Normal";} else {
          if (motor & 0x01) {err_T += "\n\t|\tMotor offline";} motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tCoil Thermal-Shutdown";} motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tMotor driver chip error";} motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tMotor bus under voltage (< 10V)";} motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tMotor bus over voltage (> 30V)";} motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tB phase sample overcurrent (> 78A)";} motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tC phase sample overcurrent (> 78A)";} motor >>= 2;
          if (motor & 0x01) {err_T += "\n\t|\tEncoder not calibrate";} motor >>= 1;
          for (int n = 1; n <= 8; n++) {
            if (motor & 0x01) {err_T += "\n\t|\tOverload error " + std::to_string(n);} motor >>= 1;
          }
          if (motor & 0x01) {err_T += "\n\t|\tA phase sample overcurrent (> 78A)";} motor >>= 7;
          if (motor & 0x01) {
            err_T += "\n\t|\tCoil Thermal too high (Over 65 Centigrade)";
          }
          motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tHall sensor calibrate error";} motor >>= 1;
          if (motor & 0x01) {err_T += "\n\t|\tTorque calibration data illegal";} motor >>= 1;
          if (motor & 0x01) {
            err_T += "\n\t|\tTorque calibration data zero offset overfit";
          }
        }
      }
      err_T += "\n\t|---------------------------------------------------";
      message_info(err_T);
    } else {
      err_T += "\n\t|---------------------------------------------------";
      message_info(err_T);
    }
  }
  return isError;
}

bool MotionManager::toml_to_lcm(const toml::table & step, trajectory_command_lcmt & traj_cmd)
{
  bool rtn_(false);
  const auto type_name = toml::get<std::string>(step.at("type"));
  const auto conf_file = local_params_dir + std::string("/orders/") + std::string("conf.toml");
  if (!std::experimental::filesystem::exists(conf_file)) {
    message_warn(std::string("Order config file not found"));
    return rtn_;
  }

  const auto conf_data = toml::parse(conf_file);
  auto params_name = toml::find<std::vector<std::string>>(conf_data, type_name);

  while (params_name.size() != 0) {
    const size_t param_size = toml::find<int>(conf_data, params_name.back());
    std::vector<float> param_val = toml::get<std::vector<float>>(step.at(params_name.back()));
    bool condition_size_valid = param_size == param_val.size();

    if (condition_size_valid) {
      traj_cmd.motionType = type_name;
      if (params_name.back() == std::string("body_cmd")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.pose_body_cmd);
      } else if (params_name.back() == std::string("contact_state")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.jump_contact);
      } else if (params_name.back() == std::string("ctrl_point")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.pose_ctrl_point);
      } else if (params_name.back() == std::string("duration")) {
        traj_cmd.duration = static_cast<int32_t>(param_val.back());
      } else if (params_name.back() == std::string("foot_cmd")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.pose_foot_cmd);
      } else if (params_name.back() == std::string("foot_support")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.pose_foot_support);
      } else if (params_name.back() == std::string("gait")) {
        traj_cmd.locomotion_gait = static_cast<int32_t>(param_val.back());
      } else if (params_name.back() == std::string("height")) {
        traj_cmd.trans_height = param_val.back();
      } else if (params_name.back() == std::string("omni")) {
        traj_cmd.locomotion_omni = static_cast<int32_t>(param_val.back());
      } else if (params_name.back() == std::string("vel")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.locomotion_vel);
      } else if (params_name.back() == std::string("w_acc_cmd")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.jump_w_acc);
      } else if (params_name.back() == std::string("x_acc_cmd")) {
        std::copy(param_val.begin(), param_val.end(), traj_cmd.jump_x_acc);
      } else {
        message_warn(std::string("Unknown config value"));
      }
    }
    params_name.pop_back();
  }

  rtn_ = true;
  return rtn_;
}
}  // namespace manager
}  // namespace cyberdog
