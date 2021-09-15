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

#include <string>
#include <iostream>
#include "xiaoai_sdk/aivs/ConnectionCapabilityImpl.h"

void ConnectionCapabilityImpl::onConnected()
{
  std::cout << "vc: ConnectionCapabilityImpl::onConnected" << std::endl;
}

void ConnectionCapabilityImpl::onDisconnected()
{
  std::cout << "vc: [ERROR]ConnectionCapabilityImpl::onDisconnected" << std::endl;
}

Network::NetworkType ConnectionCapabilityImpl::onGetNetworkType()
{
  /**
   * 根据真实情况告知sdk当前网络类型：Network::NetworkType::WIFI、Network::NetworkType::DATA、Network::NetworkType::HOTSPOT
   */
  return Network::NetworkType::WIFI;
}

void ConnectionCapabilityImpl::onGetSSID(std::string & ssid)
{
  // 提供设备当前连接的WiFi ssid，无法获取时可以不设置ssid
  ssid = "mioffice-5g";
}

void ConnectionCapabilityImpl::onLastPackageSend(std::string eventId)
{
  std::cout << "vc: ConnectionCapabilityImpl::onLastPackageSend:" << eventId << std::endl;
}
