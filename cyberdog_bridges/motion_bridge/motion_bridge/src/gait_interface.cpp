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

// C++ headers
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>  // NOLINT
#include <unordered_map>
#include <vector>
#include <utility>

#include "motion_bridge/gait_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "toml11/toml.hpp"

namespace cyberdog
{
namespace bridge
{

/* -------[Gait Mono]------- */

GaitMono::GaitMono(
  const uint8_t & gait_id,
  const std::string & gait_name,
  const std::string & gait_tag)
: gait_const_id_(gait_id),
  gait_name_(gait_name),
  gait_tag_(gait_tag),
  trans_gait_flag_(true)
{}

GaitMono::GaitMono(
  const uint8_t & gait_id,
  const std::string & gait_name,
  const std::string & gait_tag,
  const uint8_t & bridge_gait,
  const std::set<std::uint8_t> & neibor_gaits)
: gait_const_id_(gait_id),
  gait_name_(gait_name),
  gait_tag_(gait_tag),
  trans_gait_flag_(false),
  bridge_gait_(bridge_gait),
  neibor_gaits_(neibor_gaits)
{}

GaitMono::~GaitMono()
{
  neibor_gaits_.clear();
}

inline bool GaitMono::find_neibor_gaits(const uint8_t & gait_id) const
{
  bool rtn_(false);
  if (neibor_gaits_.find(gait_id) != neibor_gaits_.end()) {
    rtn_ = true;
  }
  return rtn_;
}

inline bool GaitMono::get_bridge_gait(uint8_t & bridge_gait) const
{
  bool rtn_(false);
  if (!trans_gait_flag_) {
    bridge_gait = this->bridge_gait_;
    rtn_ = true;
  }
  return rtn_;
}

/* -------[Gait Interface]------- */

bool GaitInterface::get_bridges_list(
  const uint8_t & goal_gait,
  const uint8_t & current_gait,
  std::vector<GaitMono> & bridges_list)
{
  bool rtn_(false);
  if (!init_) {
    util::message_info(
      GAIT_INTERFACE +
      std::string("Got bridges list failed. Gait interface is not initialized yet."));
  } else {
    rtn_ = get_bridge_list_internal(
      gait_bridges_max_,
      goal_gait,
      current_gait,
      gait_map_,
      bridges_list);
  }
  return rtn_;
}

bool GaitInterface::get_bridge_list_internal(
  const uint8_t & bridges_max,
  const uint8_t & goal_gait,
  const uint8_t & current_gait,
  const std::map<uint8_t, GaitMono> & gait_map,
  std::vector<GaitMono> & bridges_list)
{
  bool rtn_(false);
  uint8_t goal_(goal_gait);
  std::vector<GaitMono> gait_list;
  if (gait_map.size() == 0 || gait_map.find(goal_) == gait_map.end()) {
    return rtn_;
  }
  if (bridges_max == 0) {
    return rtn_;
  }
  while (!gait_map.at(goal_).find_neibor_gaits(current_gait)) {
    gait_list.push_back(gait_map.at(goal_));
    gait_map.at(goal_).get_bridge_gait(goal_);
    if (gait_list.size() >= bridges_max) {return rtn_;}
  }
  gait_list.push_back(gait_map.at(goal_));
  bridges_list = gait_list;
  rtn_ = true;
  return rtn_;
}

bool GaitInterface::init_gait_map(const std::string & toml_path)
{
  std::vector<toml::table> gait_bridges_tab;
  toml::table gait_define_tab;
  toml::table gait_trans_tab;
  std::unordered_map<std::string, uint8_t> gait_str_to_id;
  uint8_t gait_bridges_max(0);
  std::map<uint8_t, GaitMono> gait_map;
  std::map<uint8_t, GaitMono> trans_gait_map;
  bool rtn_(false);
  bool miss_code(false);
  toml::value toml_value;
  if (!util::get_toml_from_path(toml_path, toml_value) ||
    !util::get_raw_from_toml(toml_value, gait_tag, gait_bridges_tab) ||
    !util::get_raw_from_toml(toml_value, gait_defines, gait_define_tab) ||
    !util::get_raw_from_toml(toml_value, bridges_max, gait_bridges_max) ||
    !util::get_raw_from_toml(toml_value, gait_trans, gait_trans_tab))
  {
    util::message_info(
      GAIT_INTERFACE +
      std::string("Got toml data failed."));
    return rtn_;
  }
  if (gait_bridges_max == 0 || gait_trans_tab.size() == 0) {
    util::message_info(
      GAIT_INTERFACE +
      std::string("Gait bridges / trans gait table length is zero. Check TOML file, please."));
    return rtn_;
  }
  for (const auto & gait_tab : gait_define_tab) {
    gait_str_to_id.insert(
      std::pair<std::string, uint8_t>(
        static_cast<std::string>(gait_tab.first),
        util::toml_get_or<uint8_t>(
          gait_tab.second.as_table(), _str_id, const_gait_code, miss_code)));
  }
  for (const auto & trans_tab : gait_trans_tab) {
    auto gait_name = trans_tab.first;
    auto gait_id = util::toml_get_or<uint8_t>(
      trans_tab.second.as_table(), _str_id, const_gait_code, miss_code);
    auto gait_tag = util::toml_get_or<std::string>(
      trans_tab.second.as_table(), _str_tag, const_gait_str, miss_code);
    trans_gait_map.insert(
      std::pair<uint8_t, GaitMono>(
        gait_id,
        GaitMono(gait_id, gait_name, gait_tag)));
  }
  for (const auto & gait_bridges : gait_bridges_tab) {
    auto gait_name = util::toml_get_or<std::string>(
      gait_bridges, _str_goal_gait, const_gait_str, miss_code);
    auto gait_tab = gait_define_tab.find(gait_name);
    // Gait tab not found
    if (gait_tab == gait_define_tab.end()) {
      util::message_info(
        GAIT_INTERFACE +
        std::string("Gait defination table [") +
        gait_name +
        std::string("] not found. Break init."));
      return rtn_;
    }
    auto gait_id = util::toml_get_or<uint8_t>(
      gait_tab->second.as_table(), _str_id, const_gait_code, miss_code);
    auto gait_tag = util::toml_get_or<std::string>(
      gait_tab->second.as_table(), _str_tag, const_gait_str, miss_code);
    auto gait_bridge = util::map_get_or<std::unordered_map, std::string, uint8_t>(
      gait_str_to_id, util::toml_get_or<std::string>(
        gait_bridges, _str_bridge_gait, const_gait_str, miss_code), const_gait_code, miss_code);
    auto neibor_gaits_v = util::toml_get_or<std::vector<std::string>>(
      gait_bridges, _str_neib_gait, std::vector<std::string>(1, const_gait_str), miss_code);
    std::set<std::uint8_t> neibor_gaits_s;

    for (const auto & neibor_gait : neibor_gaits_v) {
      neibor_gaits_s.insert(
        util::map_get_or<std::unordered_map, std::string, uint8_t>(
          gait_str_to_id, neibor_gait, const_gait_code, miss_code));
    }
    gait_map.emplace(
      std::make_pair(gait_id, GaitMono(gait_id, gait_name, gait_tag, gait_bridge, neibor_gaits_s)));
  }
  // auto test
  if (miss_code) {
    util::message_info(
      GAIT_INTERFACE +
      std::string("Some configuration is missed. Check gait toml, please."));
    return rtn_;
  }
  if (gait_coherency_test(gait_bridges_max, gait_map)) {
    gait_map_ = std::move(gait_map);
    trans_gait_map_ = std::move(trans_gait_map);
    gait_bridges_max_ = std::move(gait_bridges_max);
    init_ = true;
    rtn_ = true;
  } else {
    util::message_info(
      GAIT_INTERFACE +
      std::string("Auto test failed. Please try with another gait bridges map"));
  }
  return rtn_;
}

bool GaitInterface::gait_coherency_test(
  const uint8_t & bridges_max,
  const std::map<uint8_t, GaitMono> & gait_map)
{
  bool rtn_(false);
  std::vector<uint8_t> id_list;
  if (gait_map.size() == 0) {return rtn_;}
  std::for_each(
    gait_map.begin(),
    gait_map.end(),
    [&](std::pair<uint8_t, GaitMono> const & gait_mono) {
      id_list.push_back(gait_mono.first);
    });
  uint8_t count_(0);
  while (id_list.size() > 0 && count_ == 0) {
    for (const auto & current_gait : gait_map) {
      std::vector<GaitMono> gait_list;
      if (!get_bridge_list_internal(
          bridges_max, id_list.back(),
          current_gait.first,
          gait_map,
          gait_list))
      {
        util::message_info(
          GAIT_INTERFACE +
          std::string("Failed when checking from ") +
          current_gait.second.get_gait_name() +
          std::string(" to ") +
          gait_map.at(id_list.back()).get_gait_name());
        count_++;
        break;
      }
    }
    id_list.pop_back();
  }
  if (count_ == 0) {rtn_ = true;}
  return rtn_;
}

}  // namespace bridge
}  // namespace cyberdog
