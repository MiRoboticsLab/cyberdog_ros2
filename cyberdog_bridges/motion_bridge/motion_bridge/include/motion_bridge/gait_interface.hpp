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

#ifndef MOTION_BRIDGE__GAIT_INTERFACE_HPP_
#define MOTION_BRIDGE__GAIT_INTERFACE_HPP_

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

namespace cyberdog
{
namespace bridge
{

class GaitMono
{
public:
  explicit GaitMono(
    const uint8_t & gait_id,
    const std::string & gait_name,
    const std::string & gait_tag);
  explicit GaitMono(
    const uint8_t & gait_id,
    const std::string & gait_name,
    const std::string & gait_tag,
    const uint8_t & bridge_gait,
    const std::set<std::uint8_t> & neibor_gaits);
  ~GaitMono();

  inline uint8_t get_gait_id() const {return this->gait_const_id_;}
  inline std::string get_gait_name() const {return this->gait_name_;}
  inline std::string get_gait_tag() const {return this->gait_tag_;}
  bool find_neibor_gaits(const uint8_t & gait_id) const;
  bool get_bridge_gait(uint8_t & bridge_gait) const;

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
  GaitInterface() {}
  ~GaitInterface() {}
  /**
   * @brief Get bridges list before checking gait
   * @param goal_gait goal gait to check
   * @param curent_gait current running gait
   * @param bridges_list list to return
   * @return true if got bridges list
   */
  bool get_bridges_list(
    const uint8_t & goal_gait,
    const uint8_t & current_gait,
    std::vector<GaitMono> & bridges_list);
  /**
   * @brief Initialize gait map
   * @param toml_path toml file path of gait_bridges_map
   * @return true if init succceed
   */
  bool init_gait_map(const std::string & toml_path);

private:
  /**
   * @brief Get bridges list before checking gait
   * @param goal_gait goal gait to check
   * @param curent_gait current running gait
   * @param bridges_list list to return
   * @return true if got bridges list
   */
  bool get_bridge_list_internal(
    const uint8_t & bridges_max,
    const uint8_t & goal_gait,
    const uint8_t & current_gait,
    const std::map<uint8_t, GaitMono> & gait_map,
    std::vector<GaitMono> & bridges_list);
  /**
   * @brief Auto test of gait map coherency
   * @return true if test pass
   */
  bool gait_coherency_test(
    const uint8_t & bridges_max,
    const std::map<uint8_t, GaitMono> & gait_map);

  // vars
  const std::string GAIT_INTERFACE = "[Gait_Interface] ";
  const toml::integer const_gait_code = 255;
  const toml::string const_gait_str = "gait_null";
  const std::string bridges_max = "gait_bridges_max";
  const std::string gait_tag = "gait_bridges_map";
  const std::string gait_defines = "gait_list";
  const std::string gait_trans = "trans_list";
  const std::string _str_id = "id";
  const std::string _str_tag = "tag";
  const std::string _str_goal_gait = "goal_gait";
  const std::string _str_bridge_gait = "bridge_gait";
  const std::string _str_neib_gait = "neib_gait";

  uint32_t init_;
  uint8_t gait_bridges_max_;
  std::map<uint8_t, GaitMono> trans_gait_map_;
  std::map<uint8_t, GaitMono> gait_map_;
};
}  // namespace bridge
}  // namespace cyberdog

#endif  // MOTION_BRIDGE__GAIT_INTERFACE_HPP_
