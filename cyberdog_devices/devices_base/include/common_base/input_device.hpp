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

#ifndef COMMON_BASE__INPUT_DEVICE_HPP_
#define COMMON_BASE__INPUT_DEVICE_HPP_

#include <functional>
#include <map>
#include <memory>

#include "common_base/common_type.hpp"

namespace cyberdog
{
namespace device
{
/**
 * @brief InputDevice is a template base class, designed for input devices.
 */
template<
  typename TargetT,
  typename DataT,
  typename ModeT,
  typename ArgK,
  typename ArgV,
  typename CalibT>
class InputDevice
{
public:
  ~InputDevice() {}
  /**
   * @brief Initialize input device with target devices map
   * @param targets_map uuid<uint32_t> to targets map
   * @return true if initialization succeed
   */
  virtual bool init(
    const std::map<uint32_t, TargetT> & targets_map) = 0;
  /**
   * @brief Pause input device
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
  void set_callback(std::function<void(std::shared_ptr<DataT> data)> cb) {data_cb_ = cb;}

protected:
  std::function<void(std::shared_ptr<DataT> data)> data_cb_;
  InputDevice() = default;
};
}  // namespace device
}  // namespace cyberdog

#endif  // COMMON_BASE__INPUT_DEVICE_HPP_
