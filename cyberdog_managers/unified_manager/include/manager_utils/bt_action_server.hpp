// Copyright (c) 2020 Sarthak Mittal
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

#ifndef MANAGER_UTILS__BT_ACTION_SERVER_HPP_
#define MANAGER_UTILS__BT_ACTION_SERVER_HPP_

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "manager_utils/ros_topic_logger.hpp"
#include "manager_utils/bt_engine.hpp"
#include "cyberdog_utils/action_server.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cyberdog
{
namespace bt_engine
{

/**
 * @class bt_engine::BtActionServer
 * @brief An action server that uses behavior tree to execute an action
 */
template<class ActionT, typename nodeT = rclcpp::Node>
class BtActionServer
{
public:
  using ActionServer = cyberdog_utils::ActionServer<ActionT>;

  typedef std::function<bool (typename ActionT::Goal::ConstSharedPtr)> OnGoalReceivedCallback;
  typedef std::function<void ()> OnLoopCallback;
  typedef std::function<void (typename ActionT::Goal::ConstSharedPtr)> OnPreemptCallback;
  typedef std::function<void (typename ActionT::Result::SharedPtr)> OnCompletionCallback;

  /**
   * @brief A constructor for cyberdog::bt_engine::BtActionServer class
   */
  explicit BtActionServer(
    const typename nodeT::WeakPtr node,
    const std::string & action_name,
    const std::vector<std::string> & plugin_lib_names,
    const std::string & default_bt_xml_filename,
    const uint32_t loop_duration,
    const uint32_t server_timeout,
    OnGoalReceivedCallback on_goal_received_callback,
    OnLoopCallback on_loop_callback,
    OnPreemptCallback on_preempt_callback,
    OnCompletionCallback on_completion_callback)
  : action_name_(action_name),
    default_bt_xml_filename_(default_bt_xml_filename),
    bt_loop_duration_(std::chrono::milliseconds(loop_duration)),
    bt_action_server_timeout_(std::chrono::milliseconds(server_timeout)),
    plugin_lib_names_(plugin_lib_names),
    node_weak_ptr_(node),
    on_goal_received_callback_(on_goal_received_callback),
    on_loop_callback_(on_loop_callback),
    on_preempt_callback_(on_preempt_callback),
    on_completion_callback_(on_completion_callback)
  {
    auto node_ = node_weak_ptr_.lock();
    if (!node_) {
      throw std::runtime_error{"Failed to lock node"};
    }

    node_clock_interface_ = node_->get_node_clock_interface();
    node_logging_interface_ = node_->get_node_logging_interface();
  }

  /**
   * @brief A destructor for bt_engine::BtActionServer class
   */
  ~BtActionServer() {}

  /**
   * @brief Configures member variables
   * Initializes action server for, builds behavior tree from xml file,
   * and calls user-defined onConfigure.
   * @return bool true on SUCCESS and false on FAILURE
   */
  bool on_configure(
    std::chrono::milliseconds bt_loop_duration = std::chrono::milliseconds(10),
    std::chrono::milliseconds bt_action_server_timeout = std::chrono::milliseconds(20))
  {
    auto node_ = node_weak_ptr_.lock();
    if (!node_) {
      throw std::runtime_error{"Failed to lock node"};
    }

    // Name client node after action name
    std::string client_node_name = action_name_;
    std::replace(client_node_name.begin(), client_node_name.end(), '/', '_');
    // Use suffix '_rclcpp_node' to keep parameter file consistency #1773
    auto options = rclcpp::NodeOptions().arguments(
      {"--ros-args",
        "-r",
        std::string("__node:=") + std::string(
          node_->get_name()) + client_node_name + "_extend_node",
        "--"});

    // Support for handling the topic-based goal pose from rviz
    client_node_ = std::make_shared<rclcpp::Node>("_", options);

    action_server_ = std::make_shared<ActionServer>(
      node_->get_node_base_interface(),
      node_->get_node_clock_interface(),
      node_->get_node_logging_interface(),
      node_->get_node_waitables_interface(),
      action_name_, std::bind(&BtActionServer<ActionT>::executeCallback, this));

    // Create the class that registers our custom nodes and executes the BT
    bt_ = std::make_unique<bt_engine::BehaviorTreeEngine>(plugin_lib_names_);

    // Create the blackboard that will be shared by all of the nodes in the tree
    blackboard_ = BT::Blackboard::create();

    // Put items on the blackboard
    blackboard_->set<rclcpp::Node::SharedPtr>("node", client_node_);
    blackboard_->set<std::chrono::milliseconds>("bt_loop_duration", bt_loop_duration);
    blackboard_->set<std::chrono::milliseconds>("server_timeout", bt_action_server_timeout);

    return true;
  }

  /**
   * @brief Activates action server
   * @return bool true on SUCCESS and false on FAILURE
   */
  bool on_activate()
  {
    if (!loadBehaviorTree(default_bt_xml_filename_)) {
      RCLCPP_ERROR(
        node_logging_interface_->get_logger(), "Error loading XML file: %s",
        default_bt_xml_filename_.c_str());
      return false;
    }
    action_server_->activate();
    return true;
  }

  /**
   * @brief Deactivates action server
   * @return bool true on SUCCESS and false on FAILURE
   */
  bool on_deactivate()
  {
    action_server_->deactivate();
    return true;
  }

  /**
   * @brief Resets member variables
   * @return bool true on SUCCESS and false on FAILURE
   */
  bool on_cleanup()
  {
    client_node_.reset();
    action_server_.reset();
    topic_logger_.reset();
    plugin_lib_names_.clear();
    current_bt_xml_filename_.clear();
    blackboard_.reset();
    bt_->haltAllActions(tree_.rootNode());
    // bt_->resetGrootMonitor();
    bt_.reset();
    return true;
  }

  // /**
  //  * @brief Enable (or disable) Groot monitoring of BT
  //  * @param Enable Groot monitoring
  //  * @param Publisher port
  //  * @param Server port
  //  */
  // void setGrootMonitoring(
  //   const bool enable,
  //   const unsigned publisher_port,
  //   const unsigned server_port);

  /**
   * @brief Replace current BT with another one
   * @param bt_xml_filename The file containing the new BT, uses default filename if empty
   * @return bool true if the resulting BT correspond to the one in bt_xml_filename. false
   * if something went wrong, and previous BT is maintained
   */
  bool loadBehaviorTree(const std::string & bt_xml_filename = "")
  {
    auto filename = bt_xml_filename.empty() ? default_bt_xml_filename_ : bt_xml_filename;

    if (current_bt_xml_filename_ == filename) {
      RCLCPP_DEBUG(
        node_logging_interface_->get_logger(),
        "BT will not be reloaded as the given xml is already loaded");
      return true;
    }

    // if a new tree is created, than the ZMQ Publisher must be destroyed
    // bt_->resetGrootMonitor();

    // Read the input BT XML from the specified file into a string
    std::ifstream xml_file(filename);

    if (!xml_file.good()) {
      RCLCPP_ERROR(
        node_logging_interface_->get_logger(), "Couldn't open input XML file: %s",
        filename.c_str());
      return false;
    }

    auto xml_string = std::string(
      std::istreambuf_iterator<char>(xml_file),
      std::istreambuf_iterator<char>());

    // Create the Behavior Tree from the XML input
    tree_ = bt_->createTreeFromText(xml_string, blackboard_);
    topic_logger_ = std::make_unique<RosTopicLogger>(client_node_, tree_);

    current_bt_xml_filename_ = filename;

    // Enable monitoring with Groot
    // if (enable_groot_monitoring_) {
    //   // optionally add max_msg_per_second = 25 (default) here
    //   try {
    //     bt_->addGrootMonitoring(&tree_, groot_publisher_port_, groot_server_port_);
    //     RCLCPP_INFO(
    //       logger_, "Enabling Groot monitoring for %s: %d, %d",
    //       action_name_.c_str(), groot_publisher_port_, groot_server_port_);
    //   } catch (const std::logic_error & e) {
    //     RCLCPP_ERROR(logger_, "ZMQ already enabled, Error: %s", e.what());
    //   }
    // }

    return true;
  }

  /**
   * @brief Getter function for BT Blackboard
   * @return BT::Blackboard::Ptr Shared pointer to current BT blackboard
   */
  BT::Blackboard::Ptr getBlackboard() const
  {
    return blackboard_;
  }

  /**
   * @brief Getter function for current BT XML filename
   * @return string Containing current BT XML filename
   */
  std::string getCurrentBTFilename() const
  {
    return current_bt_xml_filename_;
  }

  /**
   * @brief Wrapper function to accept pending goal if a preempt has been requested
   * @return Shared pointer to pending action goal
   */
  const std::shared_ptr<const typename ActionT::Goal> acceptPendingGoal()
  {
    return action_server_->accept_pending_goal();
  }

  /**
   * @brief Wrapper function to terminate pending goal if a preempt has been requested
   */
  void terminatePendingGoal()
  {
    action_server_->terminate_pending_goal();
  }

  /**
   * @brief Wrapper function to get current goal
   * @return Shared pointer to current action goal
   */
  const std::shared_ptr<const typename ActionT::Goal> getCurrentGoal() const
  {
    return action_server_->get_current_goal();
  }

  /**
   * @brief Wrapper function to get pending goal
   * @return Shared pointer to pending action goal
   */
  const std::shared_ptr<const typename ActionT::Goal> getPendingGoal() const
  {
    return action_server_->get_pending_goal();
  }

  /**
   * @brief Wrapper function to publish action feedback
   */
  void publishFeedback(typename std::shared_ptr<typename ActionT::Feedback> feedback)
  {
    action_server_->publish_feedback(feedback);
  }

  /**
   * @brief Getter function for the current BT tree
   * @return BT::Tree Current behavior tree
   */
  BT::Tree getTree() const
  {
    return tree_;
  }

  /**
   * @brief Function to halt the current tree. It will interrupt the execution of RUNNING nodes
   * by calling their halt() implementation (only for Async nodes that may return RUNNING)
   */
  void haltTree()
  {
    tree_.rootNode()->halt();
  }

protected:
  /**
   * @brief Action server callback
   */
  void executeCallback()
  {
    if (!on_goal_received_callback_(action_server_->get_current_goal())) {
      action_server_->terminate_current();
      return;
    }

    auto is_canceling = [&]() {
        if (action_server_ == nullptr) {
          RCLCPP_DEBUG(
            node_logging_interface_->get_logger(), "Action server unavailable. Canceling.");
          return true;
        }
        if (!action_server_->is_server_active()) {
          RCLCPP_DEBUG(
            node_logging_interface_->get_logger(), "Action server is inactive. Canceling.");
          return true;
        }
        return action_server_->is_cancel_requested();
      };

    auto on_loop = [&]() {
        if (action_server_->is_preempt_requested() && on_preempt_callback_) {
          on_preempt_callback_(action_server_->get_pending_goal());
        }
        topic_logger_->flush();
        on_loop_callback_();
      };

    // Execute the BT that was previously created in the configure step
    BtStatus rc = bt_->run(&tree_, on_loop, is_canceling, bt_loop_duration_);

    // Make sure that the Bt is not in a running state from a previous execution
    // note: if all the ControlNodes are implemented correctly, this is not needed.
    bt_->haltAllActions(tree_.rootNode());

    // Give server an opportunity to populate the result message or simple give
    // an indication that the action is complete.
    auto result = std::make_shared<typename ActionT::Result>();
    on_completion_callback_(result);

    switch (rc) {
      case BtStatus::SUCCEEDED:
        RCLCPP_INFO(node_logging_interface_->get_logger(), "Goal succeeded");
        action_server_->succeeded_current(result);
        break;

      case BtStatus::FAILED:
        RCLCPP_ERROR(node_logging_interface_->get_logger(), "Goal failed");
        action_server_->terminate_current(result);
        break;

      case BtStatus::CANCELED:
        RCLCPP_INFO(node_logging_interface_->get_logger(), "Goal canceled");
        action_server_->terminate_all(result);
        break;
    }
  }

  // Parameters for Groot monitoring
  // bool enable_groot_monitoring_ = true;
  // int groot_publisher_port_ = 1666;
  // int groot_server_port_ = 1667;

private:
  // parameters
  std::string action_name_;
  std::string default_bt_xml_filename_;
  std::string current_bt_xml_filename_;
  std::vector<std::string> plugin_lib_names_;
  typename nodeT::WeakPtr node_weak_ptr_;
  std::chrono::milliseconds bt_loop_duration_;
  std::chrono::milliseconds bt_action_server_timeout_;

  // user-provided callbacks
  OnGoalReceivedCallback on_goal_received_callback_;
  OnLoopCallback on_loop_callback_;
  OnPreemptCallback on_preempt_callback_;
  OnCompletionCallback on_completion_callback_;

  // node interfaces
  rclcpp::node_interfaces::NodeClockInterface::SharedPtr node_clock_interface_;
  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr node_logging_interface_;

  // internal variables
  std::shared_ptr<ActionServer> action_server_;
  BT::Tree tree_;
  BT::Blackboard::Ptr blackboard_;
  std::unique_ptr<BehaviorTreeEngine> bt_;
  rclcpp::Node::SharedPtr client_node_;
  std::unique_ptr<RosTopicLogger> topic_logger_;
};
}  // namespace bt_engine
}  // namespace cyberdog

#endif  // MANAGER_UTILS__BT_ACTION_SERVER_HPP_
