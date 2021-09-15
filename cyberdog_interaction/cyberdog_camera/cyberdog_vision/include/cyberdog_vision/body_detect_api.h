// Copyright (c) 2021  Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
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

#ifndef CYBERDOG_VISION__BODY_DETECT_API_H_
#define CYBERDOG_VISION__BODY_DETECT_API_H_

#include <string>
#include <vector>
#include "./algorithm.h"

/**
 * @brief  Single body info.
 **/
typedef struct
{
  std::string id;
  Rect rect;
  std::vector < float > feats;
  float score;
} SingleBodyInfo;

/**
 * @brief  Body detect algorithm initialize method.
 * @return Algorithm handle, NULL if init failed.
 **/
AlgoHandle body_detect_init();

/**
 * @brief  Destroy algorithm handle.
 * @param  Handle from init method.
 **/
void body_detect_destroy(AlgoHandle handle);

/**
 * @brief  Detect all body info in image.
 * @param  Algorithm handle.
 * @param  Input image buffer.
 * @param  Output body infos.
 * @return True if success.
 **/
bool body_detect(
  AlgoHandle handle,
  BufferInfo * buffer,
  std::vector < SingleBodyInfo > & bodies);

#endif  // CYBERDOG_VISION__BODY_DETECT_API_H_
