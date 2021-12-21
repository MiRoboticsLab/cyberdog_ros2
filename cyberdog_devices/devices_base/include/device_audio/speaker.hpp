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

#ifndef DEVICE_AUDIO__SPEAKER_HPP_
#define DEVICE_AUDIO__SPEAKER_HPP_

#include <variant>  // NOLINT
#include <vector>

#include "common_base/common_type.hpp"
#include "common_base/output_device.hpp"
#include "device_audio/audio_type.hpp"

namespace cyberdog
{
namespace device
{

struct SpeakerTargetT
{
  uint16_t id;
  uint16_t type;
  PoseT relat_pose;
};

typedef uint32_t SpeakerModeT;
typedef uint32_t SpeakerArgK;
typedef double SpeakerAmpD;
typedef bool SpeakerCalibT;

/**
 * @brief Speaker is designed for all types of audio speaker devices.
 * You will output any audio data by using same class.
 * You must initialize device with function init(), set light modules informations, and
 * synchronize saved light effects to devices. The synchronization mechanism is up
 * to devices and protocol. After initialization, you can do anything you want.
 */
class Speaker : public virtual OutputDevice
  <SpeakerTargetT, SpeakerModeT, SpeakerArgK, AudioT, SpeakerAmpD, SpeakerCalibT> {};
}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_AUDIO__SPEAKER_HPP_
