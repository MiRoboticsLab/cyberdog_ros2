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

#ifndef COMMON_BASE__OUTPUT_DEVICE_HPP_
#define COMMON_BASE__OUTPUT_DEVICE_HPP_

#include <map>

#include "common_base/common_type.hpp"

namespace cyberdog
{
namespace device
{
/**
 * @brief OutputDevice is a template base class, designed for onput devices.
 */
template<
  typename TargetT,
  typename ModeT,
  typename ArgK,
  typename ArgV,
  typename ArgD,
  typename CalibT>
class OutputDevice
{
public:
  ~OutputDevice() {}
  /**
   * @brief Initialize onput device with target devices map
   * @param targets_map UUID<uint32_t> to targets map
   * @return true if initialization succeed
   */
  virtual bool init(
    const std::map<uint32_t, TargetT> & targets_map,
    const std::map<ArgK, ArgV> &) = 0;
  /**
   * @brief Pause onput device
   * @return true if pause succeed
   */
  virtual bool pause() = 0;
  /**
   * @brief Set operating mode
   * @param target target module device
   * @param mode operating mode, enumerated in ModeT
   * @return true if set mode succeed
   */
  virtual bool set_mode(
    const TargetT & target,
    const ModeT & mode) = 0;
  /**
   * @brief Set argument key and argument value to device
   * @param target target module device
   * @param argument_key map key will be writen in device, defined in ArgK
   * @param argument_value map value will be writen in device, defined in ArgV
   * @return true if set succeed
   */
  virtual bool set_arg(
    const TargetT & target,
    const ArgK & argument_key,
    const ArgV & argument_value) = 0;
  /**
   * @brief Set calilbration data to device
   * @param target target module device
   * @param calib_data calibration data defined in CalibT
   * @return true if set succeed
   */
  virtual bool set_calib(
    const TargetT & target,
    const CalibT & calib_data) = 0;
  /**
   * @brief Get status of current device
   * @param target target module device
   * @param device_status current status of the target device, enumerated in StatusT
   * @return true if get status succeed
   */
  virtual bool get_status(
    const TargetT & target,
    StatusT & device_status) = 0;
  /**
   * @brief Run specific id set before
   * @param target target module device
   * @param argument_key map key will be writen in device, defined in ArgK
   * @param argument_description map key description will be writen in device, defined in ArgD
   * @return true if run id succeed
   */
  virtual bool send_with_key(
    const TargetT & target,
    const ArgK & argument_key,
    const ArgD & argument_description) = 0;
  /**
   * @brief Test raw data with argument value
   * @param target target module device
   * @param argument_value map value will be writen in device, defined in ArgV
   * @return true if test succeed
   */
  virtual bool send_with_value(
    const TargetT & target,
    const ArgV & argument_value) = 0;

protected:
  OutputDevice() = default;
};
}  // namespace device
}  // namespace cyberdog

#endif  // COMMON_BASE__OUTPUT_DEVICE_HPP_
