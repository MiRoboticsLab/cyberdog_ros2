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

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "fake_data.hpp"
#include "lightdisplay_base/strip_light.hpp"
#include "rclcpp/rclcpp.hpp"

namespace cyberdog
{
namespace device
{

class LightCommon : public StripLight
{
public:
  LightCommon() {}
  bool init(
    const std::vector<LightModule> & modules,
    const effects_map & saved_map = {}) override
  {
    if (modules.size() == 0) {return false;}
    if (saved_map.size() > 0) {
      if (!compare_effects(saved_map)) {
        return false;
      }
    }
    infos_ = std::move(modules);
    effects_map_ = std::move(saved_map);
    for (const auto & info : infos_) {
      light_status_.emplace(std::pair<uint16_t, uint8_t>(info.id, LightMode::PAUSE));
    }
    init_ = true;
    return true;
  }
  bool pause() override
  {
    if (!init_) {return false;}
    for (const auto & info : infos_) {
      auto value_iter = light_status_.find(info.id);
      if (value_iter != light_status_.end()) {
        value_iter->second = LightMode::PAUSE;
      }
    }
    return true;
  }
  bool set_mode(
    const LightModule & target,
    const uint8_t & light_mode) override
  {
    if (!init_) {return false;}
    auto value_iter = light_status_.find(target.id);
    if (value_iter == light_status_.end()) {
      return false;
    } else {value_iter->second = std::move(light_mode);}
    return true;
  }
  bool set_id(
    const LightModule & target,
    const uint16_t & effect_id,
    const effect_frame_v & effect_frames) override
  {
    if (!init_) {return false;}
    auto value_iter = light_status_.find(target.id);
    if (value_iter == light_status_.end()) {return false;}
    if (value_iter->second != LightMode::SET) {return false;}
    auto mcu_iter = mcu_maps.find(target.id);
    if (mcu_iter == mcu_maps.end()) {return false;}
    auto effect_iter = mcu_iter->second.find(effect_id);
    if (effect_iter != mcu_iter->second.end()) {
      mcu_iter->second.erase(effect_id);
      mcu_iter->second.emplace(std::pair<uint16_t, effect_frame_v>(effect_id, effect_frames));
    } else {
      mcu_iter->second.emplace(std::pair<uint16_t, effect_frame_v>(effect_id, effect_frames));
    }

    return true;
  }
  bool test_frames(
    const LightModule & target,
    const effect_frame_v & effect_frames) override
  {
    (void)effect_frames;
    if (!init_) {return false;}
    auto value_iter = light_status_.find(target.id);
    if (value_iter == light_status_.end()) {return false;}
    if (value_iter->second != LightMode::TEST) {return false;}
    return true;
  }
  bool run_id(
    const LightModule & target,
    const uint16_t & effect_id,
    const uint64_t & time_duration_ns) override
  {
    if (!init_) {return false;}
    (void)effect_id;
    (void)time_duration_ns;
    auto target_iter = mcu_maps.find(target.id);
    if (target_iter == mcu_maps.end()) {return false;}
    auto value_iter = target_iter->second.find(effect_id);
    if (value_iter == target_iter->second.end()) {return false;}
    auto status_iter = light_status_.find(target.id);
    if (status_iter == light_status_.end()) {return false;}
    if (status_iter->second != LightMode::RUNNING) {return false;}
    return true;
  }
  bool get_status(
    const LightModule & target,
    uint32_t & device_status) override
  {
    if (!init_) {
      device_status = LightMode::DEFAULT;
      return false;
    }
    auto value_iter = light_status_.find(target.id);
    if (value_iter == light_status_.end()) {
      return false;
    } else {device_status = std::move(value_iter->second);}
    return true;
  }

private:
  std::vector<size_t> gen_hash(const effects_map & map_to_hash)
  {
    std::hash<std::string> hash_generator;
    std::vector<size_t> hash_v;
    for (const auto & effect : map_to_hash) {
      std::string map_str;
      map_str.append(std::to_string(effect.first));
      for (const auto & frame : effect.second) {
        for (const auto & effect : frame.seq) {
          map_str.append(std::to_string(effect));
        }
        map_str.append(std::to_string(frame.effect_type));
        map_str.append(frame.effect_args);
      }
      hash_v.push_back(hash_generator(map_str));
    }
    return hash_v;
  }  // LCOV_EXCL_LINE
  bool compare_effects(const effects_map & input_map)
  {
    bool rtn_flag_(true);
    effects_map merged_map;
    for (const auto & single_mcu : mcu_maps) {
      merged_map.merge(static_cast<effects_map>(single_mcu.second));
    }
    if (gen_hash(merged_map) != gen_hash(input_map)) {return false;}
    return rtn_flag_;
  }

  bool init_;

  // MCU status
  std::map<uint16_t, uint8_t> light_status_;
  std::map<uint16_t, effects_map> mcu_maps = {
    {modules_test_real[0].id, {{1, effect_frames_single}}},
    {modules_test_real[1].id, {{2, effect_frames_multiple}}},
    {modules_test_real[2].id, {{3, effect_frames_rgb_a}, {4, effect_frames_rgb_b}}}};
};
}  // namespace device
}  // namespace cyberdog

/* Plugin */
#include "pluginlib/class_list_macros.hpp"

PLUGINLIB_EXPORT_CLASS(cyberdog::device::LightCommon, cyberdog::device::StripLight)
