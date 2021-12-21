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

#ifndef DEVICE_VISION__CAMERA_HPP_
#define DEVICE_VISION__CAMERA_HPP_

#include <array>
#include <string>
#include <variant>  // NOLINT
#include <vector>

#include "common_base/common_type.hpp"
#include "common_base/input_device.hpp"

namespace cyberdog
{
namespace device
{

struct CameraTargetT
{
  uint16_t id;
  uint16_t type;
  PoseT relat_pose;
};
struct CameraArgV
{
  std::string distortion_model;
  uint32_t height;
  uint32_t width;
};

struct CameraCalibT
{
  std::vector<double> distortion_paras;
  std::array<double, 9> intrinstic_matrix;
  std::array<double, 9> rectification_matrix;
  std::array<double, 12> projection_matrix;
};

typedef std::vector<uint8_t> Image8uT;
typedef std::vector<int8_t> Image8sT;
typedef std::vector<uint16_t> Image16uT;
typedef std::vector<int16_t> Image16sT;
typedef std::vector<int32_t> Image32sT;
typedef std::vector<float> Image32fT;
typedef std::vector<double> Image64fT;
typedef std::variant<Image8uT, Image8sT,
    Image16uT, Image16sT, Image32sT, Image32fT, Image64fT> ImageT;
typedef uint32_t CameraModeT;
typedef uint32_t CameraArgK;

/**
 * @brief Camera is designed for complementary metal-oxide semiconductor or charge-coupled device
 * cameras with uint8_t, int8_t, uint16_t, int16_t, int32_t, float32_t, double64_t image stream.
 * You will got sequential data from this device after setting callback function.
 * You must initialize device with function init(), set device modules informations, and
 * synchronize data if you need. The synchronization mechanism is up to devices and protocol.
 * After initialization, set callback function, please.
 * Argument Key to Argument Map is designed for calibration test online.
 */
class Camera : public virtual InputDevice
  <CameraTargetT, ImageT, CameraModeT, CameraArgK, CameraArgV, CameraCalibT> {};
}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_VISION__CAMERA_HPP_
