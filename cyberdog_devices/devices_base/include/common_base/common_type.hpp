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

#ifndef COMMON_BASE__COMMON_TYPE_HPP_
#define COMMON_BASE__COMMON_TYPE_HPP_

#include <cstdint>

namespace cyberdog
{
namespace device
{

struct PoseT
{
  double pos_x;
  double pos_y;
  double pos_z;
  double q_w;
  double q_x;
  double q_y;
  double q_z;
};

typedef uint32_t StatusT;
const StatusT STATUS_DEFAULT = 0x0000'0000;
}  // namespace device
}  // namespace cyberdog

#endif  // COMMON_BASE__COMMON_TYPE_HPP_
