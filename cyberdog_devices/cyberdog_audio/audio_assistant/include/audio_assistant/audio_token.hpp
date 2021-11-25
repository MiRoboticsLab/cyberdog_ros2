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

#ifndef AUDIO_ASSISTANT__AUDIO_TOKEN_HPP_
#define AUDIO_ASSISTANT__AUDIO_TOKEN_HPP_

#include <string>
#include <iostream>
#include <memory>

#include "sys/fcntl.h"
#include "sys/unistd.h"

namespace std
{
class audioToken
{
public:
  static shared_ptr<string> mToken_access;
  static shared_ptr<string> mToken_refresh;
  static shared_ptr<string> mToken_deviceid;

  audioToken();
  ~audioToken();
  std::string getAccessToken(void);
  std::string getRefreshToken(void);
  std::string getDeviceId(void);
  void updateToken(bool useTesttoken);

protected:
private:
};  // class audioToken
}  // namespace std

#endif  // AUDIO_ASSISTANT__AUDIO_TOKEN_HPP_
