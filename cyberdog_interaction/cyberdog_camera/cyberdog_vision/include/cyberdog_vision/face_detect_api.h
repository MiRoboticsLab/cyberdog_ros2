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

#ifndef CYBERDOG_VISION__FACE_DETECT_API_H_
#define CYBERDOG_VISION__FACE_DETECT_API_H_

#include <map>
#include <string>
#include <vector>
#include "./algorithm.h"

/**
 * @brief  Face id to features map.
 **/
typedef std::map < std::string, std::vector < float >> FeatureMap;

/**
 * @brief  Single face info.
 **/
typedef struct
{
  std::vector < float > lmks;
  std::vector < float > feats;
  std::vector < float > poses;
  Rect rect;
  float score;
} SingleFaceInfo;

/**
 * @brief  Single face info with matched result.
 **/
typedef struct
{
  std::string id;       // matched id in endlib
  SingleFaceInfo info;  // face info
  float match_score;    // max matched score
} MatchedFaceInfo;

/**
 * @brief  Initialize face detect AI algorithm.
 * @param
 * @return  Algorithm handle, NULL if init failed.
 **/
AlgoHandle face_detect_init(const float match_thres = 0.7f);

/**
 * @brief  Destroy face detect algorithm handle.
 * @param  Handle from init().
 **/
void face_detect_destroy(AlgoHandle handle);

/**
 * @brief  Detect all face info in image.
 * @param  handle  Algorithm handle.
 * @param  buffer  Input image buffer.
 * @param  faces   Output face infos.
 * @return Ture if success.
 **/
bool face_detect(
  AlgoHandle handle,
  BufferInfo * buffer,
  std::vector < SingleFaceInfo > & faces);

/**
 * @brief  Detect faces and compare to features endlib, get matched face info.
 * @param  handle   Algorithm handle.
 * @param  buffer   Input image buffer.
 * @param  endlibs  Faces endlib which compare to.
 * @param  matches  Output face infos with id.
 * @return True if success.
 **/
bool face_match(
  AlgoHandle handle,
  BufferInfo * buffer,
  const std::map < std::string, std::vector < float >> & endlibs,
  std::vector < MatchedFaceInfo > & matches);

#endif  // CYBERDOG_VISION__FACE_DETECT_API_H_
