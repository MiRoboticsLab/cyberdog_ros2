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

#ifndef CYBERDOG_MOTION_BRIDGE__GAIT_INTERFACE_HPP_
#define CYBERDOG_MOTION_BRIDGE__GAIT_INTERFACE_HPP_

// C++ headers
#include <map>
#include <memory>
#include <set>
#include <string>
#include <string_view>  // NOLINT
#include <vector>
#include <utility>

#include "cyberdog_utils/parser.hpp"
#include "rclcpp/rclcpp.hpp"
#include "toml11/toml.hpp"

namespace cyberdog_motion_bridge
{
class GaitMono
{
public:
  explicit GaitMono(
    const uint8_t gait_id,
    const std::string gait_name,
    const std::string gait_tag)
  : gait_const_id_(gait_id),
    gait_name_(gait_name),
    gait_tag_(gait_tag),
    trans_gait_flag_(true)
  {}
  explicit GaitMono(
    const uint8_t gait_id,
    const std::string gait_name,
    const std::string gait_tag,
    const uint8_t bridge_gait,
    const std::set<std::uint8_t> neibor_gaits)
  : gait_const_id_(gait_id),
    gait_name_(gait_name),
    gait_tag_(gait_tag),
    trans_gait_flag_(false),
    bridge_gait_(bridge_gait),
    neibor_gaits_(neibor_gaits)
  {}
  ~GaitMono()
  {
    neibor_gaits_.clear();
  }

  inline uint8_t get_gait_id() const {return this->gait_const_id_;}
  inline std::string get_gait_name() const {return this->gait_name_;}
  inline std::string get_gait_tag() const {return this->gait_tag_;}
  inline bool find_neibor_gaits(const uint8_t gait_id) const
  {
    bool rtn_(false);
    if (neibor_gaits_.find(gait_id) != neibor_gaits_.end()) {
      rtn_ = true;
    }
    return rtn_;
  }
  inline bool get_bridge_gait(uint8_t & bridge_gait) const
  {
    bool rtn_(false);
    if (!trans_gait_flag_) {
      bridge_gait = this->bridge_gait_;
      rtn_ = true;
    }
    return rtn_;
  }

private:
  uint8_t gait_const_id_;
  std::string gait_name_;
  std::string gait_tag_;
  bool trans_gait_flag_;
  uint8_t bridge_gait_;
  std::set<std::uint8_t> neibor_gaits_;
};

class GaitInterface
{
public:
  /**
   * @brief An constructor for GaitInterface
   * @param toml_path toml file path of gait bridges map
   * @param log_interface ROS log interface
   */
  explicit GaitInterface(
    const std::string toml_path,
    rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr log_interface)
  : logger_(log_interface)
  {
    if (!init_) {
      if (init_gait_map(toml_path)) {toml_path_ = std::make_shared<std::string>(toml_path);}
    }
  }

  /**
   * @brief An simple constructor for GaitInterface
   * @param log_interface ROS log interface
   */
  explicit GaitInterface(rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr log_interface)
  : logger_(log_interface)
  {
    if (!init_) {
      cyberdog_utils::message_info(
        logger_,
        GAIT_INTERFACE +
        std::string("Gait instance is empty."));
    }
  }

  ~GaitInterface()
  {
    if (!init_) {
      gait_map_.reset();
      toml_path_.reset();
    }
  }

  /**
   * @brief Get bridges list before checking gait
   * @param goal_gait goal gait to check
   * @param curent_gait current running gait
   * @param bridges_list list to return
   * @return true if got bridges list
   */
  bool get_bridges_list(
    const uint8_t goal_gait,
    const uint8_t current_gait,
    std::vector<GaitMono> & bridges_list)
  {
    bool rtn_(false);

    if (!init_) {
      cyberdog_utils::message_info(
        logger_,
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

private:
  /**
   * @brief Get bridges list before checking gait
   * @param goal_gait goal gait to check
   * @param curent_gait current running gait
   * @param bridges_list list to return
   * @return true if got bridges list
   */
  bool get_bridge_list_internal(
    const uint8_t bridges_max,
    const uint8_t goal_gait,
    const uint8_t current_gait,
    const std::shared_ptr<std::map<uint8_t, GaitMono>> gait_map,
    std::vector<GaitMono> & bridges_list)
  {
    bool rtn_(false);
    uint8_t goal_(goal_gait);
    std::vector<GaitMono> gait_list;

    if (gait_map == nullptr || gait_map->find(goal_) == gait_map->end()) {
      return rtn_;
    }
    if (bridges_max == 0) {
      return rtn_;
    }

    while (!gait_map->at(goal_).find_neibor_gaits(current_gait)) {
      gait_list.push_back(gait_map->at(goal_));
      gait_map->at(goal_).get_bridge_gait(goal_);
      if (gait_list.size() >= bridges_max) {return rtn_;}
    }

    gait_list.push_back(gait_map->at(goal_));
    bridges_list = gait_list;
    rtn_ = true;

    return rtn_;
  }
  /**
   * @brief Initialize gait map
   * @param toml_path toml file path of gait_bridges_map
   * @return true if init succceed
   */
  bool init_gait_map(const std::string toml_path)
  {
    const auto bridges_max = std::string("gait_bridges_max");
    const auto gait_tag = std::string("gait_bridges_map");
    const auto gait_defines = std::string("gait_list");
    const auto gait_trans = std::string("trans_list");
    std::vector<toml::table> gait_bridges_tab;
    toml::table gait_define_tab;
    toml::table gait_trans_tab;
    std::map<std::string, uint8_t> gait_str_to_id;
    uint8_t gait_bridges_max(0);
    auto gait_map_ptr = std::make_shared<std::map<uint8_t, GaitMono>>();
    auto trans_gait_map_ptr = std::make_shared<std::map<uint8_t, GaitMono>>();
    bool rtn_(false);

    if (!cyberdog_utils::get_raw_toml_data(toml_path, gait_tag, gait_bridges_tab) ||
      !cyberdog_utils::get_raw_toml_data(toml_path, gait_defines, gait_define_tab) ||
      !cyberdog_utils::get_raw_toml_data(toml_path, bridges_max, gait_bridges_max) ||
      !cyberdog_utils::get_raw_toml_data(toml_path, gait_trans, gait_trans_tab))
    {
      cyberdog_utils::message_info(
        logger_,
        GAIT_INTERFACE +
        std::string("Got toml data failed."));
      return rtn_;
    }

    if (gait_bridges_max == 0 || gait_trans_tab.size() == 0) {
      cyberdog_utils::message_info(
        logger_,
        GAIT_INTERFACE +
        std::string("Gait bridges / trans gait table length is zero. Check TOML file, please."));
      return rtn_;
    }

    for (const auto & gait_tab : gait_define_tab) {
      gait_str_to_id.insert(
        std::pair<std::string, uint8_t>(
          gait_tab.first,
          toml::get<std::uint8_t>(gait_tab.second.as_table().at("id"))));
    }
    for (const auto & trans_tab : gait_trans_tab) {
      auto gait_name = trans_tab.first;
      auto gait_id = toml::get<std::uint8_t>(trans_tab.second.as_table().at("id"));
      auto gait_tag = toml::get<std::string>(trans_tab.second.as_table().at("tag"));

      trans_gait_map_ptr->insert(
        std::pair<uint8_t, GaitMono>(
          gait_id,
          GaitMono(gait_id, gait_name, gait_tag)));
    }
    for (const auto & gait_bridges : gait_bridges_tab) {
      auto gait_name = toml::get<std::string>(gait_bridges.at("goal_gait"));
      auto gait_tab = gait_define_tab.find(gait_name);
      auto gait_id = toml::get<std::uint8_t>(gait_tab->second.as_table().at("id"));
      auto gait_tag = toml::get<std::string>(gait_tab->second.as_table().at("tag"));
      auto gait_bridge = gait_str_to_id.at(toml::get<std::string>(gait_bridges.at("bridge_gait")));
      auto neibor_gaits_v = toml::get<std::vector<std::string>>(gait_bridges.at("neib_gait"));
      std::set<std::uint8_t> neibor_gaits_s;

      for (const auto & neibor_gait : neibor_gaits_v) {
        neibor_gaits_s.insert(gait_str_to_id.at(neibor_gait));
      }
      auto gait_mono = GaitMono(gait_id, gait_name, gait_tag, gait_bridge, neibor_gaits_s);
      gait_map_ptr->emplace(std::make_pair(gait_id, gait_mono));
    }

    // auto test
    if (gait_coherency_test(gait_bridges_max, gait_map_ptr)) {
      gait_map_ = gait_map_ptr;
      trans_gait_map_ = trans_gait_map_ptr;
      gait_bridges_max_ = gait_bridges_max;
      init_++;
      rtn_ = true;
    } else {
      cyberdog_utils::message_info(
        logger_,
        GAIT_INTERFACE +
        std::string("Auto test failed. Please try with another gait bridges map"));
    }
    return rtn_;
  }

  /**
   * @brief Auto test of gait map coherency
   * @return true if test pass
   */
  bool gait_coherency_test(
    const uint8_t bridges_max,
    const std::shared_ptr<std::map<uint8_t, GaitMono>> gait_map)
  {
    bool rtn_(false);
    std::vector<uint8_t> id_list;

    if (gait_map == nullptr) {return rtn_;}

    std::for_each(
      gait_map->begin(),
      gait_map->end(),
      [&](std::pair<uint8_t, GaitMono> const & gait_mono) {
        id_list.push_back(gait_mono.first);
      });
    uint8_t count_(0);

    while (id_list.size() > 0 && count_ == 0) {
      for (const auto & current_gait : *gait_map) {
        std::vector<GaitMono> gait_list;
        if (!get_bridge_list_internal(
            bridges_max, id_list.back(), current_gait.first, gait_map, gait_list))
        {
          cyberdog_utils::message_info(
            logger_,
            GAIT_INTERFACE +
            std::string("Failed when checking from ") +
            current_gait.second.get_gait_name() +
            std::string(" to ") +
            gait_map->at(id_list.back()).get_gait_name());
          count_++;
          break;
        }
      }
      id_list.pop_back();
    }
    if (count_ == 0) {rtn_ = true;}
    return rtn_;
  }

  // vars
  const std::string GAIT_INTERFACE = "[Gait_Interface] ";

  inline static uint32_t init_;
  inline static uint8_t gait_bridges_max_;
  inline static std::shared_ptr<std::map<uint8_t, GaitMono>> trans_gait_map_;
  inline static std::shared_ptr<std::map<uint8_t, GaitMono>> gait_map_;
  inline static std::shared_ptr<std::string> toml_path_;

  rclcpp::node_interfaces::NodeLoggingInterface::SharedPtr logger_;
};
}  // namespace cyberdog_motion_bridge

#endif  // CYBERDOG_MOTION_BRIDGE__GAIT_INTERFACE_HPP_
