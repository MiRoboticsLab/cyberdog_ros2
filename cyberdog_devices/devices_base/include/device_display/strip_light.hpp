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

#ifndef DEVICE_DISPLAY__STRIP_LIGHT_HPP_
#define DEVICE_DISPLAY__STRIP_LIGHT_HPP_

#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "common_base/output_device.hpp"

namespace cyberdog
{
namespace device
{

struct LightTargetT
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
#define GREEN_SHIFT 8
#define BLUE_SHIFT 0
typedef uint32_t LightEffect;
typedef uint32_t LightModeT;
typedef uint32_t LightArgK;
typedef std::vector<LightFrame> LightArgV;
typedef uint32_t LightArgD_ms;
typedef std::map<LightArgK, LightArgV> LightEffectMapT;

/**
 * @brief StripLight is designed for RGB/Mono color light or light strips devices.
 * You will operate both RGB color lights and Mono color lights in same class.
 * You must initialize device with function init(), set light modules informations, and
 * synchronize saved light effects to devices. The synchronization mechanism is up
 * to devices and protocol. After initialization, you can do anything you want.
 */
class StripLight : public virtual OutputDevice
  <LightTargetT, LightModeT, LightArgK, LightArgV, LightArgD_ms, bool> {};
}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_DISPLAY__STRIP_LIGHT_HPP_
