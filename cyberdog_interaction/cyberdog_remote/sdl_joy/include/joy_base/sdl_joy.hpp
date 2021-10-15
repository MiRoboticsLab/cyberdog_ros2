// Copyright (c) 2020 Open Source Robotics Foundation.
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

#ifndef JOY_BASE__SDL_JOY_HPP_
#define JOY_BASE__SDL_JOY_HPP_

// C++ headers
#include <iostream>
#include <future>
#include <memory>
#include <string>
#include <thread>
#include <functional>

// SDL2 headers
#include "SDL2/SDL.h"

#include "sensor_msgs/msg/joy.hpp"
#include "toml11/toml.hpp"
#include "ament_index_cpp/get_package_share_directory.hpp"

namespace sdl_joy
{
using joymsg_callback = std::function<void (const sensor_msgs::msg::Joy::SharedPtr)>;
using joyconnect_callback = std::function<void (std::string)>;
using joylost_callback = std::function<void (void)>;

enum Feedback_type {action, success, failed};
class SDLJoy
{
public:
  explicit SDLJoy(
    joymsg_callback msg_callback,
    joyconnect_callback connect_callback = nullptr,
    joylost_callback lost_callback = nullptr);
  ~SDLJoy();
  void Set_feedback(Feedback_type type);
  bool IsConnect();
  std::string ConnectName();

private:
  void eventThread();
  void feedbackThread();

  bool handleJoyAxis(const SDL_Event & e);
  bool handleJoyButtonDown(const SDL_Event & e);
  bool handleJoyButtonUp(const SDL_Event & e);
  bool handleJoyHatMotion(const SDL_Event & e);
  void handleJoyDeviceAdded(const SDL_Event & e);
  void handleJoyDeviceRemoved(const SDL_Event & e);
  float convertRawAxisValueToROS(int16_t val);

  int dev_id_{0};
  int normal_callback_ms_{100};

  joymsg_callback joymsg_callback_{nullptr};
  joyconnect_callback joyconnect_callback_{nullptr};
  joylost_callback lost_callback_ {nullptr};
  SDL_Joystick * joystick_{nullptr};
  SDL_Haptic * haptic_{nullptr};
  int32_t joystick_instance_id_{0};
  double scaled_deadzone_{0.0};
  double unscaled_deadzone_{0.0};
  double scale_{0.0};
  bool sticky_buttons_{false};
  bool need_exit_{false};
  bool feedback_playing_{false};
  uint8_t feedback_cmd_list_{0};
  std::string dev_name_;
  std::thread event_thread_;
  std::thread feedback_thread_;
  std::string connect_name_;

  sensor_msgs::msg::Joy joy_msg_;
  toml::value toml_data_;
};  // end class SDLJoy
}  // end namespace sdl_joy

#endif  // JOY_BASE__SDL_JOY_HPP_
