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

#ifndef DEVICE_TOUCH__TOUCH_PAD_HPP_
#define DEVICE_TOUCH__TOUCH_PAD_HPP_

#include <variant>  // NOLINT
#include <vector>

#include "common_base/common_type.hpp"
#include "common_base/input_device.hpp"

namespace cyberdog
{
namespace device
{

struct TouchPadTargetT
{
  uint16_t id;
  uint16_t type;
  PoseT relat_pose;
};

typedef uint32_t TouchPadModeT;
typedef uint32_t TouchEventT;
typedef std::vector<bool> TouchRawDataT;
typedef std::variant<TouchEventT, TouchRawDataT> TouchDataT;
typedef bool TouchArgK;
typedef bool TouchCalibT;

/**
 * @brief TouchPad is designed for touch detection devices with event and raw positions
 *  stream data.
 * You will got sequential data from this device after setting callback function.
 * You must initialize device with function init(), set device modules informations, and
 * synchronize data if you need. The synchronization mechanism is up to devices and protocol.
 * After initialization, set callback function, please.
 * Argument Key to Argument Map is designed for calibration test online.
 */
class TouchPad : public InputDevice
  <TouchPadTargetT, TouchDataT, TouchPadModeT, TouchArgK, TouchCalibT, TouchCalibT> {};
}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_TOUCH__TOUCH_PAD_HPP_
