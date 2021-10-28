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

#ifndef GPS_PLUGINS__GPS_PLUGINS_HPP_
#define GPS_PLUGINS__GPS_PLUGINS_HPP_

#include <memory>

#include "bcm_gps/bcm_gps.hpp"
#include "gps_base/gps_base.hpp"

namespace gps_plugins
{
class Cyberdog_BCMGPS : public gps_base::Cyberdog_GPS
{
public:
  void Open() override;
  void Start() override;
  void Stop() override;
  void Close() override;

private:
  std::shared_ptr<bcm_gps::GPS> bcmgps_;
  void BCMGPS_Payload_callback(std::shared_ptr<bcm_gps::GPS_Payload> payload);
};  // class Cyberdog_BCMGPS
}  // namespace gps_plugins

#endif  // GPS_PLUGINS__GPS_PLUGINS_HPP_
