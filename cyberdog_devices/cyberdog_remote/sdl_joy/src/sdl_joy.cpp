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

#include <sys/time.h>

#include <string>
#include <chrono>
#include <memory>

#include "joy_base/sdl_joy.hpp"

#define SINGLE_DELAY 100

sdl_joy::SDLJoy::SDLJoy(
  joymsg_callback msg_callback,
  joyconnect_callback connect_callback,
  joylost_callback lost_callback)
{
  // Get package share dir
  #ifdef PACKAGE_NAME
  auto local_share_dir = ament_index_cpp::get_package_share_directory(PACKAGE_NAME);
  auto local_params_dir = local_share_dir + std::string("/params/");
  #endif
  toml_data_ = toml::parse(local_params_dir + "joy_conf.toml");

  joymsg_callback_ = msg_callback;
  joyconnect_callback_ = connect_callback;
  lost_callback_ = lost_callback;
  connect_name_ = "";

  dev_id_ = toml::find_or<int>(toml_data_, "device_id", 0);
  dev_name_ = toml::find_or<std::string>(toml_data_, "device_name", std::string(""));
  normal_callback_ms_ = toml::find_or<int>(toml_data_, "normal_callback_ms", 100);

  // The user specifies the deadzone to us in the range of 0.0 to 1.0.  Later on
  // we'll convert that to the range of 0 to 32767.  Note also that negatives
  // are not allowed, as this is a +/- value.
  scaled_deadzone_ = toml::find_or<double>(toml_data_, "deadzone", 0.05);
  if (scaled_deadzone_ < 0.0 || scaled_deadzone_ > 1.0) {
    throw std::runtime_error("Deadzone must be between 0.0 and 1.0");
  }
  unscaled_deadzone_ = 32767.0 * scaled_deadzone_;
  // According to the SDL documentation, this always returns a value between
  // -32768 and 32767.  However, we want to report a value between -1.0 and 1.0,
  // hence the "scale" dividing by 32767.  Also note that SDL returns the axes
  // with "forward" and "left" as negative.  This is opposite to the ROS
  // conventionof "forward" and "left" as positive, so we invert the axes here
  // as well.  Finally, we take into account the amount of deadzone so we truly
  // do get value between -1.0 and 1.0 (and not -deadzone to +deadzone).
  scale_ = static_cast<float>(-1.0 / (1.0 - scaled_deadzone_) / 32767.0);

  sticky_buttons_ = toml::find_or<bool>(toml_data_, "sticky_buttons", false);

  // In theory we could do this with just a timer, which would simplify the code
  // a bit.  But then we couldn't react to "immediate" events, so we stick with
  // the thread.
  event_thread_ = std::thread(&SDLJoy::eventThread, this);
}

sdl_joy::SDLJoy::~SDLJoy()
{
  need_exit_ = true;
  event_thread_.join();
  if (haptic_ != nullptr) {
    SDL_HapticClose(haptic_);
  }
  if (joystick_ != nullptr) {
    SDL_JoystickClose(joystick_);
  }
  SDL_Quit();
  std::cout << "[SDLJoy][AllExit]SDL and Mixer close\n";
}

void sdl_joy::SDLJoy::Set_feedback(Feedback_type fb_type)
{
  if (haptic_ == nullptr) {return;}
  switch (fb_type) {
    case Feedback_type::action:
      feedback_cmd_list_ = 0x1;
      break;
    case Feedback_type::success:
      feedback_cmd_list_ = 0x63;
      break;
    default:
      feedback_cmd_list_ = 0xFF;
      break;
  }
  if (feedback_playing_ == false) {
    feedback_playing_ = true;
    feedback_thread_ = std::thread(&SDLJoy::feedbackThread, this);
    feedback_thread_.detach();
  }
}

bool sdl_joy::SDLJoy::IsConnect()
{
  return joystick_ != nullptr;
}

std::string sdl_joy::SDLJoy::ConnectName()
{
  return connect_name_;
}

float sdl_joy::SDLJoy::convertRawAxisValueToROS(int16_t val)
{
  // SDL reports axis values between -32768 and 32767.  To make sure
  // we report out scaled value between -1.0 and 1.0, we add one to
  // the value iff it is exactly -32768.  This makes all of the math
  // below work properly.
  if (val == -32768) {
    val = -32767;
  }
  // Note that we do all of the math in double space below.  This ensures
  // that the values stay between -1.0 and 1.0.
  double double_val = static_cast<double>(val);
  // Apply the deadzone semantic here.  This allows the deadzone
  // to be "smooth".
  if (double_val > unscaled_deadzone_) {
    double_val -= unscaled_deadzone_;
  } else if (double_val < -unscaled_deadzone_) {
    double_val += unscaled_deadzone_;
  } else {
    double_val = 0.0;
  }
  return static_cast<float>(double_val * scale_);
}

bool sdl_joy::SDLJoy::handleJoyAxis(const SDL_Event & e)
{
  bool publish = false;

  if (e.jaxis.which != joystick_instance_id_) {
    return publish;
  }

  if (e.jaxis.axis >= joy_msg_.axes.size()) {
    std::cout << "[SDLJoy][WARN]Saw axes too large for this device, ignoring\n";
    return publish;
  }

  joy_msg_.axes.at(e.jaxis.axis) = convertRawAxisValueToROS(e.jaxis.value);

  return true;
}

bool sdl_joy::SDLJoy::handleJoyButtonDown(const SDL_Event & e)
{
  bool publish = false;

  if (e.jbutton.which != joystick_instance_id_) {
    return publish;
  }

  if (e.jbutton.button >= joy_msg_.buttons.size()) {
    std::cout << "[SDLJoy][WARN]Saw button too large for this device, ignoring\n";
    return publish;
  }

  if (sticky_buttons_) {
    // For sticky buttons, invert 0 -> 1 or 1 -> 0
    joy_msg_.buttons.at(e.jbutton.button) = 1 - joy_msg_.buttons.at(e.jbutton.button);
  } else {
    joy_msg_.buttons.at(e.jbutton.button) = 1;
  }
  publish = true;

  return publish;
}

bool sdl_joy::SDLJoy::handleJoyButtonUp(const SDL_Event & e)
{
  bool publish = false;

  if (e.jbutton.which != joystick_instance_id_) {
    return publish;
  }

  if (e.jbutton.button >= joy_msg_.buttons.size()) {
    std::cout << "[SDLJoy][WARN]Saw button too large for this device, ignoring\n";
    return publish;
  }

  if (!sticky_buttons_) {
    joy_msg_.buttons.at(e.jbutton.button) = 0;
    publish = true;
  }

  return publish;
}

bool sdl_joy::SDLJoy::handleJoyHatMotion(const SDL_Event & e)
{
  bool publish = false;

  if (e.jhat.which != joystick_instance_id_) {
    return publish;
  }

  // The hats are the last axes in the axes list.  There are two axes per hat;
  // the first of the pair is for left (positive) and right (negative), while
  // the second of the pair is for up (positive) and down (negative).

  // Determine which pair we are based on e.jhat.hat
  int num_axes = SDL_JoystickNumAxes(joystick_);
  if (num_axes < 0) {
    std::cout << "[SDLJoy][WARN]Failed to get axes: " << SDL_GetError() << "\n";
    return publish;
  }
  size_t axes_start_index = num_axes + e.jhat.hat * 2;
  // Note that we check axes_start_index + 1 here to ensure that we can write to
  // either the left/right axis or the up/down axis that corresponds to this hat.
  if ((axes_start_index + 1) >= joy_msg_.axes.size()) {
    std::cout << "[SDLJoy][WARN]Saw hat too large for this device, ignoring\n";
    return publish;
  }

  if (e.jhat.value & SDL_HAT_LEFT) {
    joy_msg_.axes.at(axes_start_index) = 1.0;
  }
  if (e.jhat.value & SDL_HAT_RIGHT) {
    joy_msg_.axes.at(axes_start_index) = -1.0;
  }
  if (e.jhat.value & SDL_HAT_UP) {
    joy_msg_.axes.at(axes_start_index + 1) = 1.0;
  }
  if (e.jhat.value & SDL_HAT_DOWN) {
    joy_msg_.axes.at(axes_start_index + 1) = -1.0;
  }
  if (e.jhat.value == SDL_HAT_CENTERED) {
    joy_msg_.axes.at(axes_start_index) = 0.0;
    joy_msg_.axes.at(axes_start_index + 1) = 0.0;
  }
  publish = true;

  return publish;
}

void sdl_joy::SDLJoy::handleJoyDeviceAdded(const SDL_Event & e)
{
  if (!dev_name_.empty()) {
    int num_joysticks = SDL_NumJoysticks();
    if (num_joysticks < 0) {
      std::cout << "[SDLJoy][WARN]Failed to get the number of joysticks: " << SDL_GetError() <<
        "\n";
      return;
    }
    for (int i = 0; i < num_joysticks; ++i) {
      const char * name = SDL_JoystickNameForIndex(i);
      if (name == nullptr) {
        std::cout << "[SDLJoy][WARN]Could not get joystick name: " << SDL_GetError() << "\n";
        continue;
      }
      if (std::string(name) == dev_name_) {
        // We found it!
        dev_id_ = i;
        break;
      }
    }
  }

  if (e.jdevice.which != dev_id_) {
    return;
  }

  joystick_ = SDL_JoystickOpen(dev_id_);
  if (joystick_ == nullptr) {
    std::cout << "[SDLJoy][WARN]Unable to open joystick " << dev_id_ << ": " << SDL_GetError() <<
      "\n";
    return;
  }

  // We need to hold onto this so that we can properly remove it on a
  // remove event.
  joystick_instance_id_ = SDL_JoystickGetDeviceInstanceID(dev_id_);
  if (joystick_instance_id_ < 0) {
    std::cout << "[SDLJoy][WARN]Failed to get instance ID for joystick: " << SDL_GetError() << "\n";
    SDL_JoystickClose(joystick_);
    joystick_ = nullptr;
    return;
  }

  int num_buttons = SDL_JoystickNumButtons(joystick_);
  if (num_buttons < 0) {
    std::cout << "[SDLJoy][WARN]Failed to get number of buttons: " << SDL_GetError() << "\n";
    SDL_JoystickClose(joystick_);
    joystick_ = nullptr;
    return;
  }
  joy_msg_.buttons.resize(num_buttons);

  int num_axes = SDL_JoystickNumAxes(joystick_);
  if (num_axes < 0) {
    std::cout << "[SDLJoy][WARN]Failed to get number of axes: " << SDL_GetError() << "\n";
    SDL_JoystickClose(joystick_);
    joystick_ = nullptr;
    return;
  }
  int num_hats = SDL_JoystickNumHats(joystick_);
  if (num_hats < 0) {
    std::cout << "[SDLJoy][WARN]Failed to get number of hats: " << SDL_GetError() << "\n";
    SDL_JoystickClose(joystick_);
    joystick_ = nullptr;
    return;
  }
  joy_msg_.axes.resize(num_axes + num_hats * 2);

  // Get the initial state for each of the axes
  for (int i = 0; i < num_axes; ++i) {
    int16_t state;
    if (SDL_JoystickGetAxisInitialState(joystick_, i, &state)) {
      joy_msg_.axes.at(i) = convertRawAxisValueToROS(state);
    }
  }

  haptic_ = SDL_HapticOpenFromJoystick(joystick_);
  if (haptic_ != nullptr) {
    if (SDL_HapticRumbleInit(haptic_) < 0) {
      // Failed to init haptic.  Clean up haptic_.
      SDL_HapticClose(haptic_);
      haptic_ = nullptr;
    }
  } else {
    std::cout << "[SDLJoy][ERROR]No haptic (rumble) available, skipping initialization\n";
  }

  connect_name_ = SDL_JoystickName(joystick_);
  std::cout << "[SDLJoy][INFO]Opened joystick: " << connect_name_ <<
    ".  deadzone: " << scaled_deadzone_ << "\n";
  if (joyconnect_callback_ != nullptr) {joyconnect_callback_(connect_name_);}
}

void sdl_joy::SDLJoy::handleJoyDeviceRemoved(const SDL_Event & e)
{
  if (e.jdevice.which != joystick_instance_id_) {
    return;
  }

  joy_msg_.buttons.resize(0);
  joy_msg_.axes.resize(0);
  if (haptic_ != nullptr) {
    SDL_HapticClose(haptic_);
    haptic_ = nullptr;
  }
  if (joystick_ != nullptr) {
    SDL_JoystickClose(joystick_);
    joystick_ = nullptr;
  }
  if (lost_callback_ != nullptr) {lost_callback_();}
}

void sdl_joy::SDLJoy::eventThread()
{
  std::cout << "[SDLJoy][INFO]Main thread start\n";
  if (SDL_Init(SDL_INIT_JOYSTICK | SDL_INIT_HAPTIC) < 0) {
    throw std::runtime_error("SDL could not be initialized: " + std::string(SDL_GetError()));
  } else {std::cout << "[SDLJoy][INFO]SDL init success\n";}

  SDL_Event e;
  do {
    if (SDL_WaitEventTimeout(&e, normal_callback_ms_) == 1) {
      // Succeeded getting an event
      if (e.type == SDL_JOYAXISMOTION) {
        handleJoyAxis(e);
      } else if (e.type == SDL_JOYBUTTONDOWN) {
        handleJoyButtonDown(e);
      } else if (e.type == SDL_JOYBUTTONUP) {
        handleJoyButtonUp(e);
      } else if (e.type == SDL_JOYHATMOTION) {
        handleJoyHatMotion(e);
      } else if (e.type == SDL_JOYDEVICEADDED) {
        handleJoyDeviceAdded(e);
      } else if (e.type == SDL_JOYDEVICEREMOVED) {
        handleJoyDeviceRemoved(e);
      }
    }

    if (joystick_ != nullptr) {
      auto sys_time = std::chrono::nanoseconds(std::chrono::system_clock::now().time_since_epoch());
      builtin_interfaces::msg::Time stamp;
      stamp.sec =
        std::chrono::duration_cast<std::chrono::seconds>(sys_time).count();
      stamp.nanosec = (sys_time -
        std::chrono::duration_cast<std::chrono::seconds>(sys_time)).count();

      joy_msg_.header.stamp = stamp;
      joy_msg_.header.frame_id = "joy";

      if (joymsg_callback_ != nullptr) {
        joymsg_callback_(std::make_shared<sensor_msgs::msg::Joy>(joy_msg_));
      }
    }
  } while (!need_exit_);
  std::cout << "[SDLJoy][INFO]Main thread exit\n";
}

void sdl_joy::SDLJoy::feedbackThread()
{
  while (feedback_cmd_list_ != 0 && !need_exit_) {
    if (haptic_ != nullptr) {SDL_HapticRumblePlay(haptic_, feedback_cmd_list_ & 0x1, 1000);}
    feedback_cmd_list_ >>= 1;
    SDL_Delay(SINGLE_DELAY);
  }
  if (haptic_ != nullptr) {SDL_HapticRumblePlay(haptic_, 0, 100);}
  feedback_playing_ = false;
}
