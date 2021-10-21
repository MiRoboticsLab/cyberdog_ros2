// Copyright (c) 2014 Clearpath Robotics, Inc., All rights reserved.
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

#include <map>
#include <utility>
#include <functional>
#include <memory>
#include <string>
#include <cmath>
#include <vector>
#include <fstream>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp_components/register_node_macro.hpp"
#include "rcutils/logging_macros.h"
#include "sensor_msgs/msg/joy.hpp"
#include "sensor_msgs/msg/joy_feedback.hpp"

#include "cyberdog_joy/cyberdog_joy.hpp"

#include "cyberdog_utils/Enums.hpp"
#include "motion_msgs/action/change_gait.hpp"
#include "motion_msgs/action/change_mode.hpp"
#include "motion_msgs/action/ext_mon_order.hpp"
#include "motion_msgs/msg/mon_order.hpp"
#include "motion_msgs/msg/frameid.hpp"
#include "motion_msgs/msg/se3_velocity_cmd.hpp"

#include "joy_base/sdl_joy.hpp"

#include "toml11/toml.hpp"

#define BUTTON_NUM 16
#define AXIS_NUM 4

#define LOCOMOTION_BRIDGE_PKG "cyberdog_motion_bridge"
#define MONO_ORDER_FILE "mono_order.toml"
#define MAIN_CONFIG_FILE "cyberdog_conf.toml"

namespace cyberdog_joy
{
using msg_joy = sensor_msgs::msg::Joy;
using msg_SE3VelocityCMD = motion_msgs::msg::SE3VelocityCMD;
using msg_MonOrder = motion_msgs::msg::MonOrder;

class remap_rule
{
public:
  char type;
  uint8_t pos;
  int8_t down;
};

enum JoyA {RV, RH, LV, LH};
enum JoyB {Main_U, Main_D, Main_L, Main_R,
  Cross_U, Cross_D, Cross_L, Cross_R,
  Trig_R, Trig_RS, Trig_L, Trig_LS,
  Menu_R, Menu_L, Stick_R, Stick_L};
std::map<std::string, int> key_id{
  {"Main_U", 0},
  {"Main_D", 1},
  {"Main_L", 2},
  {"Main_R", 3},
  {"Cross_U", 4},
  {"Cross_D", 5},
  {"Cross_L", 6},
  {"Cross_R", 7},
  {"Trig_R", 8},
  {"Trig_RS", 9},
  {"Trig_L", 10},
  {"Trig_LS", 11},
  {"Menu_R", 12},
  {"Menu_L", 13},
  {"Stick_R", 14},
  {"Stick_L", 15}
};

struct CyberdogJoy::Impl
{
  void failedUnlock();
  void showSelfdata();
  void showRawdata(const msg_joy::SharedPtr joy_msg);
  void joyCallback(const msg_joy::SharedPtr joy_msg);
  void nextRemapStep();
  void remapping(const msg_joy::SharedPtr joy_msg);
  void sendMonoOrder(int page, int button);
  bool get4B();
  bool getB(std::string func_key);
  bool getB(int button_id);
  float getA(int axis_id);
  void showRemapping();
  void saveRemapping();
  bool loadRules(std::string fileName);
  void joyconnectCallback(std::string joy_name);
  void joylostCallback();
  void mainEvent(int deltarT);

  void check_gait_sync(const uint8_t & gait_id);
  void check_mode_sync(const uint8_t & mode_id);
  bool check_action_sync(const uint8_t & order_id, const double & para = 0);

  rclcpp::Publisher<msg_SE3VelocityCMD>::SharedPtr cmd_vel_pub;

  rclcpp_action::Client<motion_msgs::action::ChangeGait>::SharedPtr gait_client_;
  rclcpp_action::Client<motion_msgs::action::ChangeMode>::SharedPtr mode_client_;
  rclcpp_action::Client<motion_msgs::action::ExtMonOrder>::SharedPtr order_client_;

  std::shared_ptr<sdl_joy::SDLJoy> Joy;
  msg_joy::SharedPtr self_joymsg;

  uint8_t run_count;

  int rulesA[AXIS_NUM];
  remap_rule rulesB[BUTTON_NUM];

  int unlock_steps;
  int min_callback_ms;
  int remapping_time_count;
  int remapping_step;

  bool unlock_state;
  bool need_unlock_everytime;
  bool enable_remapping;
  bool show_raw_joydata;
  bool show_self_joydata;
  bool show_joyconfig;
  bool enable_feedback_cmd;
  bool enable_muilt_page;
  bool require_enable_button;

  float unlock_max;
  float unlock_min;

  double last_callback;

  std::string local_params_dir;
  std::string locomotion_params_dir;
  std::string joyconfig_file;

  toml::table order_map;
  toml::table func_key_map;
  std::vector<double> scale_linear;
  std::vector<double> scale_angular;
  std::vector<int> tmp_axis;
  std::vector<toml::table> action_order_list;

  rclcpp::Clock::SharedPtr clock_interface;
  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr logger_interface;
};

/**
 * Constructs CyberdogJoy.
 */
CyberdogJoy::CyberdogJoy(const rclcpp::NodeOptions & options)
: Node("cyberdog_joy_node", options)
{
  pimpl_ = new Impl;
  // Get package share dir
  #ifdef PACKAGE_NAME
  auto local_share_dir = ament_index_cpp::get_package_share_directory(PACKAGE_NAME);
  pimpl_->local_params_dir = local_share_dir + std::string("/params/");
  #endif
  auto locomotion_bridge_dir = ament_index_cpp::get_package_share_directory(LOCOMOTION_BRIDGE_PKG);
  pimpl_->locomotion_params_dir = locomotion_bridge_dir + std::string("/params/");
  auto config_toml = toml::parse(pimpl_->local_params_dir + MAIN_CONFIG_FILE);

  pimpl_->Joy =
    std::make_shared<sdl_joy::SDLJoy>(
    std::bind(
      &CyberdogJoy::Impl::joyCallback, this->pimpl_,
      std::placeholders::_1),
    std::bind(
      &CyberdogJoy::Impl::joyconnectCallback, this->pimpl_,
      std::placeholders::_1),
    std::bind(
      &CyberdogJoy::Impl::joylostCallback, this->pimpl_)
    );

  pimpl_->remapping_time_count = 0;
  pimpl_->last_callback = 0;
  pimpl_->enable_remapping = false;
  pimpl_->unlock_steps = 0;
  bool need_unlock = toml::find_or<bool>(config_toml, "need_unlock", true);
  pimpl_->unlock_state = !need_unlock;
  if (need_unlock) {
    pimpl_->need_unlock_everytime = toml::find_or<bool>(config_toml, "need_unlock_everytime", true);
  } else {pimpl_->need_unlock_everytime = false;}
  pimpl_->show_raw_joydata = toml::find_or<bool>(config_toml, "show_raw_joydata", false);
  pimpl_->show_self_joydata = toml::find_or<bool>(config_toml, "show_self_joydata", false);
  pimpl_->min_callback_ms = toml::find_or<int>(config_toml, "min_callback_ms", 10);
  pimpl_->show_joyconfig = toml::find_or<bool>(config_toml, "show_joyconfig", false);
  pimpl_->require_enable_button = toml::find_or<bool>(config_toml, "require_enable_button", true);
  pimpl_->enable_muilt_page = toml::find_or<bool>(config_toml, "enable_muilt_page", false);
  pimpl_->unlock_max = toml::find_or<float>(config_toml, "unlock_max", 0.6);
  pimpl_->unlock_min = toml::find_or<float>(config_toml, "unlock_min", 0.4);
  pimpl_->action_order_list = toml::find<std::vector<toml::table>>(config_toml, "action_order");
  pimpl_->func_key_map = toml::find<toml::table>(config_toml, "func_key");

  std::string load_para = toml::find_or<std::string>(config_toml, "load_para", "normal_para");
  toml::table para_table = toml::find<toml::table>(config_toml, load_para);
  pimpl_->scale_linear = toml::get<std::vector<double>>(para_table.at("scale_linear"));
  pimpl_->scale_angular = toml::get<std::vector<double>>(para_table.at("scale_angular"));
  printf(
    "[JoyCtrl][INFO]Notice:\n\tscale_linear[%lf,%lf,%lf]\n\tscale_angular[%lf,%lf,%lf]\n",
    pimpl_->scale_linear[0], pimpl_->scale_linear[1], pimpl_->scale_linear[2],
    pimpl_->scale_angular[0], pimpl_->scale_angular[1], pimpl_->scale_angular[2]
  );

  pimpl_->run_count = 6;
  auto order_toml = toml::parse(pimpl_->locomotion_params_dir + MONO_ORDER_FILE);
  pimpl_->order_map = toml::find<toml::table>(order_toml, "order_list");

  pimpl_->cmd_vel_pub = this->create_publisher<msg_SE3VelocityCMD>(
    "body_cmd",
    rclcpp::SystemDefaultsQoS());
  pimpl_->gait_client_ = rclcpp_action::create_client<motion_msgs::action::ChangeGait>(
    this,
    "checkout_gait");
  pimpl_->mode_client_ = rclcpp_action::create_client<motion_msgs::action::ChangeMode>(
    this,
    "checkout_mode");
  pimpl_->order_client_ = rclcpp_action::create_client<motion_msgs::action::ExtMonOrder>(
    this,
    "exe_monorder");

  pimpl_->clock_interface = this->get_clock();
  pimpl_->logger_interface = this->get_node_logging_interface();
}

CyberdogJoy::~CyberdogJoy()
{
  delete pimpl_;
}

void CyberdogJoy::Impl::failedUnlock()
{
  if (unlock_steps == 0) {return;}
  RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Other button or axes get, reset unlock step\n");
  Joy->Set_feedback(sdl_joy::Feedback_type::failed);
  unlock_steps = 0;
}

void CyberdogJoy::Impl::showSelfdata()
{
  std::cout << "axes:[";
  for (int a = 0; a < AXIS_NUM; a++) {
    std::cout << getA(a);
    if (a != AXIS_NUM - 1) {std::cout << ", ";}
  }
  std::cout << "]\n";
  std::cout << "button:[";
  for (int a = 0; a < BUTTON_NUM; a++) {
    std::cout << getB(a);
    if (a != BUTTON_NUM - 1) {std::cout << ", ";}
  }
  std::cout << "]\n";
}

void CyberdogJoy::Impl::showRawdata(const msg_joy::SharedPtr joy_msg)
{
  std::cout << "frame_id:" << joy_msg->header.frame_id << "\n";
  std::cout << "stamp:" << joy_msg->header.stamp.sec << ".";
  printf("%09d\n", joy_msg->header.stamp.nanosec);
  int num = static_cast<int>(joy_msg->buttons.size());
  std::cout << "button:[";
  for (int a = 0; a < num; a++) {
    std::cout << joy_msg->buttons[a];
    if (a != num - 1) {std::cout << ", ";}
  }
  std::cout << "]\n";
  num = static_cast<int>(joy_msg->axes.size());
  std::cout << "axes:[";
  for (int a = 0; a < num; a++) {
    std::cout << joy_msg->axes[a];
    if (a != num - 1) {std::cout << ", ";}
  }
  std::cout << "]\n";
}

void CyberdogJoy::Impl::nextRemapStep()
{
  remapping_step++;
  Joy->Set_feedback(sdl_joy::Feedback_type::action);
}

void CyberdogJoy::Impl::remapping(const msg_joy::SharedPtr joy_msg)
{
  static bool wait_reset = false;
  int button_num = static_cast<int>(joy_msg->buttons.size());
  int axis_num = static_cast<int>(joy_msg->axes.size());

  if (wait_reset == true) {
    wait_reset = false;
    for (int a = 0; a < button_num; a++) {
      if (joy_msg->buttons[a] != 0) {wait_reset = true;}
    }
    for (int a = 0; a < axis_num && a < static_cast<int>(tmp_axis.size()); a++) {
      if (joy_msg->axes[a] != tmp_axis[a]) {wait_reset = true;}
    }
  }

  if (remapping_step == -1) {
    for (int a = 0; a < button_num; a++) {
      if (joy_msg->buttons[a] != 0) {return;}
    }
    tmp_axis.clear();
    tmp_axis = std::vector<int>(axis_num);
    wait_reset = false;
    float axis_def_val;
    for (int a = 0; a < axis_num; a++) {
      axis_def_val = joy_msg->axes[a];
      if (axis_def_val != 0 && axis_def_val != 1 && axis_def_val != -1) {
        printf("[Joy][remapping][WARN]Axis[%d] may not default value\n", a);
      }
      tmp_axis[a] = static_cast<int>(axis_def_val);
    }
    remapping_step++;
    return;
  } else if (wait_reset == false) {
    if (remapping_step < BUTTON_NUM) {
      for (int a = 0; a < button_num; a++) {
        if (joy_msg->buttons[a] == 1) {
          std::cout << "[Joy][remapping][INFO]GetButton remapping to button\n";
          wait_reset = true;
          rulesB[remapping_step].type = 'B';
          rulesB[remapping_step].pos = a;
          rulesB[remapping_step].down = 0;
          nextRemapStep();
          return;
        }
      }
      for (int a = 0; a < axis_num && a < static_cast<int>(tmp_axis.size()); a++) {
        if (joy_msg->axes[a] != tmp_axis[a] && abs(joy_msg->axes[a]) == 1) {
          std::cout << "[Joy][remapping][INFO]GetAxis remapping to button\n";
          wait_reset = true;
          rulesB[remapping_step].type = 'A';
          rulesB[remapping_step].pos = a;
          rulesB[remapping_step].down = static_cast<int>(joy_msg->axes[a]);
          nextRemapStep();
          return;
        }
      }
    } else if (remapping_step - BUTTON_NUM < AXIS_NUM) {
      for (int a = 0; a < axis_num && a < static_cast<int>(tmp_axis.size()); a++) {
        if (joy_msg->axes[a] != tmp_axis[a] && joy_msg->axes[a] == 1) {
          std::cout << "[Joy][remapping][INFO]GetAxis remapping\n";
          wait_reset = true;
          rulesA[remapping_step - BUTTON_NUM] = a;
          nextRemapStep();
        }
      }
    } else {
      std::cout << "[Joy][remapping][INFO]Finish remapping\n";
      Joy->Set_feedback(sdl_joy::Feedback_type::success);
      showRemapping();
      saveRemapping();
      enable_remapping = false;
    }
  }
}

void CyberdogJoy::Impl::sendMonoOrder(int page, int button)
{
  if (page < static_cast<int>(action_order_list.size())) {
    std::string action_name = "action" + std::to_string(button);
    if (action_order_list[page].count(action_name) != 0) {
      std::string order_name = toml::get<std::string>(action_order_list[page].at(action_name));
      if (order_name == "") {return;}

      double para = 0;
      std::string para_name = action_name + "_para";
      if (action_order_list[page].count(para_name) != 0) {
        para = toml::get<double>(action_order_list[page].at(para_name));
      }

      if (order_map.count(order_name) != 0) {
        printf("[Joy][INFO]Send mono_order name:[%s]\n", order_name.c_str());
        int order_id = toml::get<int>(order_map.at(order_name));
        check_action_sync(order_id, para);
      } else {
        printf(
          "[Joy][ERROR]Cant find mono_order name:[%s] in %s\n",
          order_name.c_str(), MONO_ORDER_FILE);
      }
    } else {
      printf(
        "[Joy][WARN]Cant find action%d in mono_order[%d] at %s\n", button, page,
        MAIN_CONFIG_FILE);
    }
  } else {
    printf("[Joy][ERROR]Cant find mono_order[%d] in %s\n", page, MAIN_CONFIG_FILE);
  }
}

bool CyberdogJoy::Impl::get4B()
{
  int num = 0;
  for (int a = 0; a < static_cast<int>(self_joymsg->buttons.size()); a++) {
    if (self_joymsg->buttons[a] != 0 && ++num >= 4) {return true;}
  }
  return false;
}

bool CyberdogJoy::Impl::getB(std::string func_key)
{
  std::string key_id_name;
  if (func_key_map.count(func_key) != 0) {
    key_id_name = toml::get<std::string>(func_key_map.at(func_key));
  } else {
    std::cout << "[Joy][ERROR]Get func_key error, no func_key named:" << func_key << "\n";
    return false;
  }

  if (key_id.count(key_id_name) != 0) {
    return getB(key_id[key_id_name]);
  }
  std::cout << "[Joy][ERROR]Get key_id_name error, no key_id named:" << key_id_name << "\n";
  return false;
}

bool CyberdogJoy::Impl::getB(int button_id)
{
  if (self_joymsg == nullptr) {return false;}
  if (button_id < BUTTON_NUM) {
    int pos = rulesB[button_id].pos;
    char type = rulesB[button_id].type;
    if (type == 'A') {
      if (pos < static_cast<int>(self_joymsg->axes.size())) {
        if (self_joymsg->axes[pos] == rulesB[button_id].down) {return true;} else {return false;}
      } else {
        std::cout <<
          "[Joy][ERROR]Get button error, over self_joymsg->axes.size, may need remapping\n";
        return false;
      }
    } else if (type == 'B') {
      if (pos < static_cast<int>(self_joymsg->buttons.size())) {
        return self_joymsg->buttons[pos] == 1;
      } else {
        std::cout <<
          "[Joy][ERROR]Get button error, over self_joymsg->buttons.size, may need remapping\n";
        return false;
      }
    } else {
      std::cout << "[Joy][ERROR]Get button error, wrong type get: " << type << "\n";
      return false;
    }
  }
  std::cout << "[Joy][ERROR]Get button error, over BUTTON_NUM";
  return false;
}

float CyberdogJoy::Impl::getA(int axis_id)
{
  if (self_joymsg == nullptr) {return 0;}
  if (axis_id < static_cast<int>(self_joymsg->axes.size())) {
    return self_joymsg->axes[rulesA[axis_id]];
  }
  std::cout << "[Joy][ERROR]Get axis error, over self_joymsg->axes.size\n";
  return 0;
}

void CyberdogJoy::Impl::showRemapping()
{
  printf(
    "[Joy][INFO]Loaded reamapping config:\n\taxis[%d, %d, %d, %d]\n",
    rulesA[0], rulesA[1], rulesA[2], rulesA[3]);
  for (int a = 0; a < BUTTON_NUM; a++) {
    printf(
      "\tbutton[%2d]: type=%c, pos=%2d, down=%2d\n", a, rulesB[a].type, rulesB[a].pos,
      rulesB[a].down);
  }
}

void CyberdogJoy::Impl::saveRemapping()
{
  try {
    FILE * fp1;
    std::string path = local_params_dir + "joys/" + joyconfig_file;
    if ((fp1 = fopen(path.c_str(), "w+")) != NULL) {
      fprintf(fp1, "axis = [");
      for (int a = 0; a < AXIS_NUM; a++) {
        fprintf(fp1, "%d", rulesA[a]);
        if (a != AXIS_NUM - 1) {fprintf(fp1, ", ");}
      }
      fprintf(fp1, "]\n\n");
      for (int a = 0; a < BUTTON_NUM; a++) {
        fprintf(fp1, "[[button]] #%d\n", a);
        fprintf(fp1, "type = \"%c\"\n", rulesB[a].type);
        fprintf(fp1, "pos = %d\n", rulesB[a].pos);
        fprintf(fp1, "down = %d\n", rulesB[a].down);
      }
      fclose(fp1);
      std::cout << "[Joy][INFO]Write file[" << joyconfig_file << "]\n";
      return;
    }
  } catch (...) {
  }
  std::cout << "[Joy][ERROR]Cant write file[" << joyconfig_file << "]\n";
}

bool CyberdogJoy::Impl::loadRules(std::string fileName)
{
  std::cout << "[Joy][INFO]Try to load joy config: " << fileName << "\n";
  try {
    auto remap_config = toml::parse(local_params_dir + "joys/" + fileName);
    auto button = toml::find<std::vector<toml::table>>(remap_config, "button");
    auto axis = toml::find<std::vector<int>>(remap_config, "axis");
    for (int a = 0; a < AXIS_NUM && a < static_cast<int>(axis.size()); a++) {
      rulesA[a] = axis[a];
    }
    for (int a = 0; a < BUTTON_NUM && a < static_cast<int>(button.size()); a++) {
      if (button[a].count("type") != 0) {
        rulesB[a].type = toml::get<std::string>(button[a].at("type"))[0];
      } else {rulesB[a].type = 'B';}
      if (button[a].count("pos") != 0) {
        rulesB[a].pos = toml::get<int>(button[a].at("pos"));
      } else {rulesB[a].pos = 0;}
      if (button[a].count("down") != 0) {
        rulesB[a].down = toml::get<int>(button[a].at("down"));
      } else {rulesB[a].down = 0;}
    }
    if (show_joyconfig) {showRemapping();}
    std::cout << "[Joy][INFO]Finish load joy config\n";
    return true;
  } catch (...) {
    std::cout << "[Joy][ERROR]Cant load joy config\n";
  }
  return false;
}

void CyberdogJoy::Impl::joyconnectCallback(std::string joy_name)
{
  joyconfig_file = joy_name;
  for (int a = 0; a < static_cast<int>(joyconfig_file.length()); a++) {
    char ch = joyconfig_file[a];
    if (!(('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z') || ('0' <= ch && ch <= '9'))) {
      joyconfig_file[a] = '_';
    }
  }
  joyconfig_file += ".toml";
  if (loadRules(joyconfig_file) == false) {loadRules("Default.toml");}
}

void CyberdogJoy::Impl::joylostCallback()
{
  if (need_unlock_everytime) {
    unlock_state = false;
    unlock_steps = 0;
  }
  std::cout << "[Joy][INFO]Joy lost connect\n";
}

void CyberdogJoy::Impl::joyCallback(const msg_joy::SharedPtr joy_msg)
{
  double now = joy_msg->header.stamp.sec +
    static_cast<double>(joy_msg->header.stamp.nanosec * 0.000000001);
  if (last_callback == 0) {last_callback = now;}
  int deltarT = (now - last_callback) * 1000;
  if (deltarT < min_callback_ms) {return;}
  last_callback = now;
  if (show_raw_joydata) {showRawdata(joy_msg);}
  if (enable_remapping) {
    remapping(joy_msg);
    return;
  }

  self_joymsg = joy_msg;
  mainEvent(deltarT);
}

void CyberdogJoy::Impl::mainEvent(int deltarT)
{
  if (show_self_joydata) {showSelfdata();}
  if (unlock_state == false) {
    if (get4B()) {remapping_time_count += deltarT;} else {remapping_time_count = 0;}
    if (remapping_time_count > 3000) {
      RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Start remapping");
      enable_remapping = true;
      remapping_step = -1;
      Joy->Set_feedback(sdl_joy::Feedback_type::action);
    }
    int downnum = 0;
    for (int a = 0; a < BUTTON_NUM; a++) {
      if (getB(a)) {downnum++;}
    }
    if (downnum != 0) {
      failedUnlock();
      return;
    }

    float LH = getA(JoyA::LH);
    float LV = getA(JoyA::LV);
    float RH = getA(JoyA::RH);
    float RV = getA(JoyA::RV);
    switch (unlock_steps) {
      case 0:
        if (LV > unlock_max && abs(LH) < unlock_min && RV > unlock_max && abs(RH) < unlock_min) {
          Joy->Set_feedback(sdl_joy::Feedback_type::action);
          unlock_steps++;
          RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Trying unlock, step:%d", unlock_steps);
        }
        break;
      case 1:
        if (abs(LV) < unlock_min && LH > unlock_max && abs(RV) < unlock_min && RH < -unlock_max) {
          Joy->Set_feedback(sdl_joy::Feedback_type::action);
          unlock_steps++;
          RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Trying unlock, step:%d", unlock_steps);
        }
        break;
      case 2:
        if (LV < -unlock_max && abs(LH) < unlock_min && RV < -unlock_max && abs(RH) < unlock_min) {
          Joy->Set_feedback(sdl_joy::Feedback_type::action);
          unlock_steps++;
          RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Trying unlock, step:%d", unlock_steps);
        }
        break;
      case 3:
        if (abs(LV) < unlock_min && LH < -unlock_max && abs(RV) < unlock_min && RH > unlock_max) {
          Joy->Set_feedback(sdl_joy::Feedback_type::action);
          unlock_steps++;
          RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Trying unlock, step:%d", unlock_steps);
        }
        break;
      case 4:
        if (LV > unlock_max && abs(LH) < unlock_min && RV > unlock_max && abs(RH) < unlock_min) {
          Joy->Set_feedback(sdl_joy::Feedback_type::action);
          unlock_steps++;
          RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Trying unlock, step:%d", unlock_steps);
        }
        break;
      case 5:
        if (abs(LV) < unlock_min && abs(LH) < unlock_min && abs(RV) < unlock_min &&
          abs(RH) < unlock_min)
        {
          unlock_state = true;
          Joy->Set_feedback(sdl_joy::Feedback_type::success);
          RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Joy control unlock success");
        }
        break;
    }
    if (unlock_state == false) {return;}
  }

  if (getB(JoyB::Stick_R) && getB(JoyB::Stick_L)) {
    RCLCPP_INFO(logger_interface->get_logger(), "[Joy]Joy control lock");
    Joy->Set_feedback(sdl_joy::Feedback_type::action);
    unlock_state = false;
    unlock_steps = 0;
    return;
  }

  static bool sent_disable_msg = false;
  if (!require_enable_button || getB("enable_control")) {
    // Initializes with zeros by default.
    auto velocity_cmd = std::make_unique<msg_SE3VelocityCMD>();

    velocity_cmd->sourceid = msg_SE3VelocityCMD::INTERNAL;
    velocity_cmd->velocity.frameid.id = motion_msgs::msg::Frameid::BODY_FRAME;
    velocity_cmd->velocity.timestamp = clock_interface->now();
    velocity_cmd->velocity.linear_x = getA(JoyA::LV) * scale_linear[0];  // 1
    velocity_cmd->velocity.linear_y = getA(JoyA::RH) * scale_linear[1];  // 3
    velocity_cmd->velocity.linear_z = 0;
    velocity_cmd->velocity.angular_z = getA(JoyA::LH) * scale_angular[1];  // yaw
    velocity_cmd->velocity.angular_y = getA(JoyA::RV) * scale_angular[0];  // pitch
    velocity_cmd->velocity.angular_x = 0;  // roll

    cmd_vel_pub->publish(std::move(velocity_cmd));
    sent_disable_msg = false;
  } else {
    // When enable button is released, immediately send a single no-motion command
    // in order to stop the robot.
    if (!sent_disable_msg) {
      // Initializes with zeros by default.
      auto velocity_cmd = std::make_unique<msg_SE3VelocityCMD>();
      velocity_cmd->sourceid = msg_SE3VelocityCMD::INTERNAL;
      velocity_cmd->velocity.frameid.id = motion_msgs::msg::Frameid::BODY_FRAME;
      velocity_cmd->velocity.timestamp = clock_interface->now();
      cmd_vel_pub->publish(std::move(velocity_cmd));
      sent_disable_msg = true;
    }
  }

  // MOD
  #define MOD_BUTTON_NUM 5
  int now_modB[MOD_BUTTON_NUM];
  static int old_modB[MOD_BUTTON_NUM] = {0};
  now_modB[0] = getB(JoyB::Cross_U);
  now_modB[1] = getB(JoyB::Cross_D);
  now_modB[2] = getB(JoyB::Cross_L);
  now_modB[3] = getB(JoyB::Cross_R);
  now_modB[4] = getB(JoyB::Menu_R);
  if (now_modB[4] > old_modB[4]) {
    check_mode_sync(motion_msgs::msg::Mode::MODE_LOCK);
    RCLCPP_INFO(logger_interface->get_logger(), "Change to lock mode");
  } else if (now_modB[0] > old_modB[0]) {
    check_mode_sync(motion_msgs::msg::Mode::MODE_MANUAL);
    RCLCPP_INFO(logger_interface->get_logger(), "Change to manual mode");
  } else if (now_modB[1] > old_modB[1]) {
    check_mode_sync(motion_msgs::msg::Mode::MODE_DEFAULT);
    RCLCPP_INFO(logger_interface->get_logger(), "Change to default mode");
  } else if (now_modB[2] > old_modB[2]) {
    if (run_count > motion_msgs::msg::Gait::GAIT_STAND_R) {
      run_count -= 1;
    }
    RCLCPP_INFO(logger_interface->get_logger(), "Change to last run gait [%d]", run_count);
    check_gait_sync(run_count);
  } else if (now_modB[3] > old_modB[3]) {
    if (run_count < motion_msgs::msg::Gait::GAIT_PRONK) {
      run_count += 1;
    }
    RCLCPP_INFO(logger_interface->get_logger(), "Change to next run gait [%d]", run_count);
    check_gait_sync(run_count);
  }
  for (int a = 0; a < MOD_BUTTON_NUM; a++) {
    old_modB[a] = now_modB[a];
  }

  // order
  #define ORDER_BUTTON_NUM 7
  int now_orderB[ORDER_BUTTON_NUM];
  static int old_orderB[ORDER_BUTTON_NUM] = {0};
  static int page = 0;
  for (int a = 0; a < 4; a++) {
    now_orderB[a] = getB("action" + std::to_string(a));
  }
  if (enable_muilt_page) {
    now_orderB[4] = getB("next_page");
    now_orderB[5] = getB("last_page");
    now_orderB[6] = getB("reset_page");
    if (now_orderB[4] > old_orderB[4]) {
      if (page + 1 < static_cast<int>(action_order_list.size())) {page++;}
    } else if (now_orderB[5] > old_orderB[5]) {
      if (page - 1 >= 0) {page--;}
    } else if (now_orderB[6] > old_orderB[6]) {page = 0;}
  } else {
    if (getB("last_page")) {page = 0;} else if (getB("next_page")) {page = 2;} else {
      page = 1;
    }
  }

  for (int a = 0; a < 4; a++) {
    if (now_orderB[a] > old_orderB[a]) {
      sendMonoOrder(page, a);
      break;
    }
  }
  for (int a = 0; a < ORDER_BUTTON_NUM; a++) {
    old_orderB[a] = now_orderB[a];
  }
}

void CyberdogJoy::Impl::check_gait_sync(const uint8_t & gait_id)
{
  Joy->Set_feedback(sdl_joy::Feedback_type::action);
  auto goal = motion_msgs::action::ChangeGait::Goal();

  goal.motivation = cyberdog_utils::GAIT_TRIG;
  goal.gaitstamped.timestamp = clock_interface->now();
  goal.gaitstamped.gait = gait_id;

  auto goal_handle = gait_client_->async_send_goal(goal);
}

void CyberdogJoy::Impl::check_mode_sync(const uint8_t & mode_id)
{
  Joy->Set_feedback(sdl_joy::Feedback_type::action);
  auto goal = motion_msgs::action::ChangeMode::Goal();

  goal.modestamped.timestamp = clock_interface->now();
  goal.modestamped.control_mode = mode_id;

  auto send_goal_options =
    rclcpp_action::Client<motion_msgs::action::ChangeMode>::SendGoalOptions();
  send_goal_options.result_callback =
    [&](const rclcpp_action::ClientGoalHandle<motion_msgs::action::ChangeMode>::WrappedResult &
      result) {
      if (result.result->succeed) {
        Joy->Set_feedback(sdl_joy::Feedback_type::success);
      } else {Joy->Set_feedback(sdl_joy::Feedback_type::failed);}
    };
  auto goal_handle = mode_client_->async_send_goal(goal, send_goal_options);
}

bool CyberdogJoy::Impl::check_action_sync(const uint8_t & order_id, const double & para)
{
  Joy->Set_feedback(sdl_joy::Feedback_type::action);
  auto goal = motion_msgs::action::ExtMonOrder::Goal();

  goal.orderstamped.timestamp = clock_interface->now();
  goal.orderstamped.id = order_id;
  goal.orderstamped.para = para;

  auto send_goal_options =
    rclcpp_action::Client<motion_msgs::action::ExtMonOrder>::SendGoalOptions();
  send_goal_options.result_callback =
    [&](const rclcpp_action::ClientGoalHandle<motion_msgs::action::ExtMonOrder>::WrappedResult &
      result) {
      if (result.result->succeed) {
        Joy->Set_feedback(sdl_joy::Feedback_type::success);
      } else {Joy->Set_feedback(sdl_joy::Feedback_type::failed);}
    };
  auto goal_handle = order_client_->async_send_goal(goal, send_goal_options);
  return true;
}

}  // namespace cyberdog_joy

RCLCPP_COMPONENTS_REGISTER_NODE(cyberdog_joy::CyberdogJoy)
