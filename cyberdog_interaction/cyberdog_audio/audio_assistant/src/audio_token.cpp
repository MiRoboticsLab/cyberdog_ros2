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
#include <memory>

#include "audio_assistant/audio_token.hpp"
#include "toml11/toml.hpp"

namespace std
{
shared_ptr<string> audioToken::mToken_access = make_shared<string>("invalid access token");
shared_ptr<string> audioToken::mToken_refresh = make_shared<string>("invalid refresh token");
shared_ptr<string> audioToken::mToken_deviceid = make_shared<string>("invalid device id");

audioToken::audioToken()
{
  std::cout << "Creating audioToken." << std::endl;
}

audioToken::~audioToken()
{
  std::cout << "Destroying audioToken" << std::endl;
}

std::string audioToken::getAccessToken(void)
{
  std::cout << "audioToken::getAccessToken() " <<
    " access: " <<
    mToken_access->c_str() <<
    std::endl;
  return *mToken_access;
}

std::string audioToken::getRefreshToken(void)
{
  std::cout << "audioToken::getRefreshToken() " <<
    " refresh: " <<
    mToken_refresh->c_str() <<
    std::endl;
  return *mToken_refresh;
}

std::string audioToken::getDeviceId(void)
{
  std::cout << "audioToken::getDeviceId() " <<
    " deviceid: " <<
    mToken_deviceid->c_str() <<
    std::endl;
  return *mToken_deviceid;
}

void audioToken::updateToken(bool useTesttoken)
{
  std::string token_access = "inlalid";
  std::string token_refresh = "inlalid";
  std::string token_deviceid = "inlalid";

  if (useTesttoken) {
    const auto tokenData = toml::parse("/opt/ros2/cyberdog/data/audio_debug.toml");

    token_access = toml::find<std::string>(tokenData, "testtoken_access");
    token_refresh = toml::find<std::string>(tokenData, "testtoken_refresh");
    token_deviceid = toml::find<std::string>(tokenData, "testtoken_deviceid");
  } else {
    const auto tokenData = toml::parse("/opt/ros2/cyberdog/data/token.toml");

    token_access = toml::find<std::string>(tokenData, "token_access");
    token_refresh = toml::find<std::string>(tokenData, "token_fresh");
    // token_deviceid = toml::find<std::string>(tokenData, "token_deviceid");
  }

  mToken_access = make_shared<string>(token_access.c_str());
  mToken_refresh = make_shared<string>(token_refresh.c_str());
  mToken_deviceid = make_shared<string>(token_deviceid.c_str());

  std::cout << "audioToken::updateToken() " <<
    " access: " <<
    mToken_access->c_str() <<
    " refresh: " <<
    mToken_refresh->c_str() <<
    " deviceid: " <<
    mToken_deviceid->c_str() <<
    std::endl;
}
}  // namespace std
