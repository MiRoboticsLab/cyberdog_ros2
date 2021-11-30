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

#ifndef AUDIO_INTERACTION__MD5_HPP_
#define AUDIO_INTERACTION__MD5_HPP_

#include <string.h>

#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <memory>

#include "openssl/md5.h"

std::string md5_cal(std::string token_s);

#endif  // AUDIO_INTERACTION__MD5_HPP_
