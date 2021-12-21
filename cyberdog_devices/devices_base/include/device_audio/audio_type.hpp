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

#ifndef DEVICE_AUDIO__AUDIO_TYPE_HPP_
#define DEVICE_AUDIO__AUDIO_TYPE_HPP_

#include <variant>  // NOLINT
#include <vector>

#include "common_base/common_type.hpp"

namespace cyberdog
{
namespace device
{

typedef std::vector<uint8_t> Audio8uT;
typedef std::vector<int16_t> Audio16sT;
typedef std::variant<Audio8uT, Audio16sT> AudioT;

}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_AUDIO__AUDIO_TYPE_HPP_
