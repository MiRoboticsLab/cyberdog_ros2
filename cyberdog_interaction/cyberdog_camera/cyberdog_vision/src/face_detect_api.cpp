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

#include <string>
#include <map>
#include <vector>
#include "vision_sdk_vendor/XMFaceAPI.h"
#include "vision_sdk_vendor/model_path.h"
#include "cyberdog_vision/face_detect_api.h"

AlgoHandle face_detect_init(const float match_thres)
{
  XMFaceAPI * api = NULL;

  std::string path = get_ai_model_path();
  std::string detect_mf = path + "/face/detect/mnetv2_gray_nop_light_epoch_235_512.onnx";
  std::string lmk_mf = path + "/face/landmark/pfldd.md";
  std::string feat_mf = path + "/face/feature/mask_mfn_v5_1_nobn.onnx";

  api = XMFaceAPI::Create();
  if (api == NULL) {
    return NULL;
  }

  if (!api->init(detect_mf, lmk_mf, feat_mf, match_thres)) {
    printf("Failed to init face api.\n");
    XMFaceAPI::Destroy(api);
    return NULL;
  }

  /* First frame takes too long time, so we run it when initializing. */
  XMImage img;
  int width = 640;
  int height = 480;
  void * buffer = malloc(width * height * 3);
  img.data = (unsigned char *)buffer;
  img.width = width;
  img.height = height;
  img.channel = 3;
  img.type = ColorType::BGR;

  std::map<std::string, std::vector<float>> feats;
  std::vector<MatchFaceInfo> faces_info;
  api->getMatchInfo(img, feats, faces_info);
  free(buffer);

  return reinterpret_cast<AlgoHandle>(api);
}

void face_detect_destroy(AlgoHandle handle)
{
  XMFaceAPI * api = reinterpret_cast<XMFaceAPI *>(handle);

  if (NULL != api) {
    XMFaceAPI::Destroy(api);
  }
}

bool face_detect(
  AlgoHandle handle,
  BufferInfo * buffer,
  std::vector<SingleFaceInfo> & faces)
{
  std::vector<EntryFaceInfo> entries;
  XMFaceAPI * api = reinterpret_cast<XMFaceAPI *>(handle);

  if (NULL == api) {
    return false;
  }

  XMImage img;
  img.data = (unsigned char *)buffer->data;
  img.width = buffer->width;
  img.height = buffer->height;
  img.channel = 3;

  switch (buffer->format) {
    case FORMAT_BGR:
      img.type = ColorType::BGR;
      break;
    case FORMAT_RGB:
      img.type = ColorType::RGB;
      break;
    default:
      return false;
  }

  if (!api->getFaceInfo(img, entries)) {
    return false;
  }

  size_t size = entries.size();
  if (size > 0) {
    faces.resize(size);
    for (uint32_t i = 0; i < entries.size(); i++) {
      faces[i].lmks = entries[i].lmks;
      faces[i].feats = entries[i].feats;
      faces[i].poses = entries[i].poses;
      faces[i].rect.left = entries[i].rect.left;
      faces[i].rect.top = entries[i].rect.top;
      faces[i].rect.width = entries[i].rect.right - entries[i].rect.left;
      faces[i].rect.height = entries[i].rect.bottom - entries[i].rect.top;
      faces[i].score = entries[i].score;
    }
  }

  return true;
}

bool face_match(
  AlgoHandle handle,
  BufferInfo * buffer,
  const std::map<std::string, std::vector<float>> & endlibs,
  std::vector<MatchedFaceInfo> & matches)
{
  std::vector<MatchFaceInfo> entries;
  XMFaceAPI * api = reinterpret_cast<XMFaceAPI *>(handle);

  if (NULL == api) {
    return false;
  }

  XMImage img;
  img.data = (unsigned char *)buffer->data;
  img.width = buffer->width;
  img.height = buffer->height;
  img.channel = 3;

  switch (buffer->format) {
    case FORMAT_BGR:
      img.type = ColorType::BGR;
      break;
    case FORMAT_RGB:
      img.type = ColorType::RGB;
      break;
    default:
      return false;
  }

  if (!api->getMatchInfo(img, endlibs, entries)) {
    return false;
  }

  size_t size = entries.size();
  if (size > 0) {
    matches.resize(size);
    for (uint32_t i = 0; i < entries.size(); i++) {
      matches[i].id = entries[i].face_id;
      matches[i].info.lmks = entries[i].lmks;
      matches[i].info.poses = entries[i].poses;
      matches[i].info.rect.left = entries[i].rect.left;
      matches[i].info.rect.top = entries[i].rect.top;
      matches[i].info.rect.width = entries[i].rect.right - entries[i].rect.left;
      matches[i].info.rect.height = entries[i].rect.bottom - entries[i].rect.top;
      matches[i].info.score = entries[i].score;
      matches[i].match_score = entries[i].match_score;
    }
  }

  return true;
}
