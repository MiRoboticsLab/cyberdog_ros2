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

#ifndef DEVICE_BALANCE__IMU_HPP_
#define DEVICE_BALANCE__IMU_HPP_

#include <variant>  // NOLINT

#include "common_base/common_type.hpp"
#include "common_base/input_device.hpp"

namespace cyberdog
{
namespace device
{

struct IMUTargetT
{
  uint16_t id;
  uint8_t type;
  uint8_t calibrated;
  PoseT relat_pose;
};

struct IMURawDataT
{
  double acc_x;
  double acc_y;
  double acc_z;
  double gyro_x;
  double gyro_y;
  double gyro_z;
  double mag_x;
  double mag_y;
  double mag_z;
};

struct IMUEulerDataT
{
  double pitch;
  double roll;
  double yaw;
};

struct IMUQuatDataT
{
  double q_w;
  double q_x;
  double q_y;
  double q_z;
};

struct AccCalibT
{
  double angle_yz;
  double angle_zy;
  double angle_zx;
  double scale_x;
  double scale_y;
  double scale_z;
  double bias_x;
  double bias_y;
  double bias_z;
};

struct GyroCalibT
{
  double angle_yz;
  double angle_zy;
  double angle_xz;
  double angle_zx;
  double angle_xy;
  double angle_yx;
  double scale_x;
  double scale_y;
  double scale_z;
};

struct MagCalibT
{
  double offset_x;
  double offset_y;
  double offset_z;
};

struct IMUCalibT
{
  AccCalibT acc_calib_data;
  GyroCalibT gyro_calib_data;
  MagCalibT mag_calib_data;
};

typedef std::variant<IMURawDataT, IMUEulerDataT, IMUQuatDataT> IMUDataT;
typedef uint32_t IMUModeT;
typedef uint32_t IMUArgK;

/**
 * @brief IMU is designed for inertial measurement unit devices with raw data, euler angle data,
 * and quaternion data.
 * You will got sequential data from this device after setting callback function.
 * You must initialize device with function init(), set device modules informations, and
 * synchronize data if you need. The synchronization mechanism is up to devices and protocol.
 * After initialization, set callback function, please.
 * Argument Key to Argument Map is designed for calibration test online.
 */
class IMU : public virtual InputDevice
  <IMUTargetT, IMUDataT, IMUModeT, IMUArgK, IMUCalibT, IMUCalibT> {};
}  // namespace device
}  // namespace cyberdog

#endif  // DEVICE_BALANCE__IMU_HPP_
