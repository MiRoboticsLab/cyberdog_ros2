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
#include <vector>
#include "vision_sdk_vendor/ContentMotionAPI.h"
#include "vision_sdk_vendor/model_path.h"
#include "cyberdog_vision/body_detect_api.h"

AlgoHandle body_detect_init()
{
  ContentMotionAPI * api = NULL;

  std::string path = get_ai_model_path();
  std::string model_path = path + "/body/detect.prototxt";
  std::string weights_path = path + "/body/detect.caffemodel";
  std::string modelcls_path = path + "/body/cls_human_mid.onnx";

  api = new ContentMotionAPI();
  if (NULL == api) {
    return NULL;
  }

  if (0 != api->Init(model_path, weights_path, modelcls_path, 0 /*gpu id*/)) {
    delete api;
    return NULL;
  }

  /* First frame takes too long time, so we run it when initializing. */
  XMImage img;
  struct LogInfo log_info;
  vector<HumanBodyInfo> res;
  int width = 640;
  int height = 480;
  void * buffer = malloc(width * height * 3);

  img.data = (unsigned char *)buffer;
  img.width = width;
  img.height = height;
  img.type = ColorType::BGR;
  api->GetContentMotionAnalyse(img, res, log_info, 0);
  free(buffer);

  return reinterpret_cast<AlgoHandle>(api);
}

void body_detect_destroy(AlgoHandle handle)
{
  if (NULL != handle) {
    ContentMotionAPI * api = reinterpret_cast<ContentMotionAPI *>(handle);
    api->Close();
    delete api;
  }
}

bool body_detect(AlgoHandle handle, BufferInfo * buffer, std::vector<SingleBodyInfo> & bodies)
{
  ContentMotionAPI * api = reinterpret_cast<ContentMotionAPI *>(handle);
  std::vector<HumanBodyInfo> entries;

  if (NULL == api || NULL == buffer || NULL == buffer->data) {
    return false;
  }

  XMImage img;
  img.data = (unsigned char *)buffer->data;
  img.width = buffer->width;
  img.height = buffer->height;

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

  struct LogInfo log_info;
  if (0 != api->GetContentMotionAnalyse(img, entries, log_info, 0 /*gpu id*/)) {
    return false;
  }

  size_t size = entries.size();
  if (size > 0) {
    bodies.resize(size);
    for (uint32_t i = 0; i < entries.size(); i++) {
      bodies[i].id = entries[i].BodyId;
      bodies[i].rect.left = entries[i].left;
      bodies[i].rect.top = entries[i].top;
      bodies[i].rect.width = entries[i].width;
      bodies[i].rect.height = entries[i].height;
      bodies[i].feats = entries[i].feats;
      bodies[i].score = entries[i].score;
    }
  }

  return true;
}
