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

#ifndef LIGHTDISPLAY_BASE__LED_BASE_HPP_
#define LIGHTDISPLAY_BASE__LED_BASE_HPP_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cyberdog
{
namespace device
{

struct RGB
{
  uint8_t red;
  uint8_t green;
  uint8_t blue;
  uint8_t reserve;
};

struct Mono
{
  uint32_t level;
};

union LightEffect {
  RGB rgb_effect;
  Mono mono_effect;
};

struct LightModule
{
  uint16_t id;
  uint16_t length;
  std::string info;
};

enum LightMode
{
  PAUSE = 0,
  RUNNING = 1,
  SET = 2,
  TEST = 4
};

struct LightFrame
{
  std::vector<LightEffect> seq;
  uint16_t effect_type;
  std::string effect_args;
};

/**
 * @brief StripLight is designed for RGB/Mono color light or light strips devices.
 * You will operate both RGB color lights and Mono color lights in same class.
 * There is no common construct function for the class.
 * You must create instance in function init(), set light modules informations, and
 * synchronize saved light effects to devices. The synchronization mechanism is up
 * to devices and protocol. After initialization, you can do anything you want.
 */
class StripLight
{
public:
  StripLight(const StripLight &) = delete;

  /**
   * @brief Initialize LED class and set parameters
   * @param modules vector of LED light modules
   * @param saved_map id to frames map saved before, sync it during init
   * @return true if initialization succeed
   */
  virtual bool init(
    const std::vector<LightModule> & modules,
    const std::map<uint16_t, std::vector<LightFrame>> & saved_map = {}) = 0;
  /**
   * @brief Pause LED
   * @return true if pause succeed
   */
  virtual bool pause() = 0;
  /**
   * @brief Set operating mode
   * @param target target module device
   * @param light_mode operating mode, enumerated in LightMode
   * @return true if set mode succeed
   */
  virtual bool set_mode(
    const LightModule & target,
    const uint8_t & light_mode) = 0;
  /**
   * @brief Set effect id to device
   * @param target target module device
   * @param effect_id effect id to set
   * @param effect_frames effect frame sequence
   * @return true if set succeed
   */
  virtual bool set_id(
    const LightModule & target,
    const uint16_t & effect_id,
    const std::vector<LightFrame> & effect_frames) = 0;
  /**
   * @brief Test raw data frame sequence
   * @param target target module device
   * @param effect_frames effect frame sequence
   * @return true if test succeed
   */
  virtual bool test_frames(
    const LightModule & target,
    const std::vector<LightFrame> & effect_frames) = 0;
  /**
   * @brief Run specific id set before
   * @param target target module device
   * @param effect_id effect id set before
   * @param time_duration_ns duration of the effect
   * @return true if run id succeed
   */
  virtual bool run_id(
    const LightModule & target,
    const uint16_t & effect_id,
    const uint64_t & time_duration_ns) = 0;
  /**
   * @brief Get status of current device
   * @param target target module device
   * @param device_status current status of the target device, enumerated in xxxx(tbd)
   * @return true if get status succeed
   */
  virtual bool get_status(
    const LightModule & target,
    uint32_t & device_status) = 0;

private:
  std::vector<LightModule> infos_;
  std::map<uint16_t, LightModule> effects_map_;
};
}  // device
}  // cyberdog

#endif  // LIGHTDISPLAY_BASE__LED_BASE_HPP_
