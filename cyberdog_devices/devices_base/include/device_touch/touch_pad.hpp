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

#ifndef DEVICE_TOUCH__TOUCH_HPP_
#define DEVICE_TOUCH__TOUCH_HPP_

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
class TouchPad
{};
}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_TOUCH__TOUCH_HPP_
