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

#ifndef LIGHTDISPLAY_BASE__STRIP_LIGHT_HPP_
#define LIGHTDISPLAY_BASE__STRIP_LIGHT_HPP_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace cyberdog
{
namespace device
{

struct LightModule
{
  uint16_t id;
  uint16_t length;
  std::string info;
};

enum LightMode
{
  DEFAULT = 0,
  PAUSE = 1,
  RUNNING = 2,
  SET = 3,
  TEST = 4
};

struct LightFrame
{
  std::vector<uint32_t> seq;
  uint16_t effect_type;
  std::string effect_args;
};
#define RED_SHIFT 16
#define GREEN_SHFIT 8
#define BLUE_SHIFT 0
typedef uint32_t LightEffect;
typedef uint16_t effect_id_t;
typedef std::vector<LightFrame> effect_frame_v;
typedef std::map<effect_id_t, effect_frame_v> effects_map;

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
  ~StripLight() {}
  /**
   * @brief Initialize LED class and set parameters
   * @param modules vector of LED light modules
   * @param saved_map id to frames map saved before, sync it during init
   * @return true if initialization succeed
   */
  virtual bool init(
    const std::vector<LightModule> & modules,
    const effects_map & saved_map = {}) = 0;
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
    const effect_frame_v & effect_frames) = 0;
  /**
   * @brief Test raw data frame sequence
   * @param target target module device
   * @param effect_frames effect frame sequence
   * @return true if test succeed
   */
  virtual bool test_frames(
    const LightModule & target,
    const effect_frame_v & effect_frames) = 0;
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

protected:
  StripLight() {}
  std::vector<LightModule> infos_;
  effects_map effects_map_;
};
}  // namespace device
}  // namespace cyberdog

#endif  // LIGHTDISPLAY_BASE__STRIP_LIGHT_HPP_
