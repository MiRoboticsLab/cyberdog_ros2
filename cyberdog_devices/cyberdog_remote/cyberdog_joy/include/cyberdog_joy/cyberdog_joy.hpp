// Copyright (c) 2014 Clearpath Robotics, Inc., All rights reserved.
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

#ifndef CYBERDOG_JOY__CYBERDOG_JOY_HPP_
#define CYBERDOG_JOY__CYBERDOG_JOY_HPP_

#include <rclcpp/rclcpp.hpp>

namespace cyberdog_joy
{
class CyberdogJoy : public rclcpp::Node
{
public:
  explicit CyberdogJoy(const rclcpp::NodeOptions & options);

  virtual ~CyberdogJoy();

private:
  struct Impl;
  Impl * pimpl_;
};

}  // namespace cyberdog_joy

#endif  // CYBERDOG_JOY__CYBERDOG_JOY_HPP_
