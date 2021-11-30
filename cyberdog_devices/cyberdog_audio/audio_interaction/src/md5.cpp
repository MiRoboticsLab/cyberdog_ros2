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

#include "audio_interaction/md5.hpp"

std::string md5_cal(std::string token_s)
{
  int len = token_s.length();
  auto token_c = new char[len + 1];
  snprintf(token_c, len + 1, "%s", token_s.c_str());
  unsigned char md[16];
  int i;
  char tmp[3] = {};
  char buf[33] = {};
  MD5((unsigned char *)token_c, strlen(token_c), md);
  for (i = 0; i < 16; i++) {
    snprintf(tmp, sizeof(tmp), "%2.2x", md[i]);
    strncat(buf, tmp, sizeof(buf));
  }
  printf("%s/n", buf);
  delete[] token_c;
  return buf;
}
