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

#include <string.h>
#include <string>
#include "vision_sdk_vendor/ReIDToolAPI.h"
#include "vision_sdk_vendor/model_path.h"
#include "cyberdog_vision/reid_api.h"

AlgoHandle reid_init()
{
  REIDHandle api = NULL;

  std::string path = get_ai_model_path();
  std::string model_path = path + "/body/reid_v1_mid.engine";

  REID_Init(api, model_path.c_str(), 0);

  return reinterpret_cast<AlgoHandle>(api);
}

void reid_destroy(AlgoHandle handle)
{
  REIDHandle api = reinterpret_cast<REIDHandle>(handle);

  if (NULL != api) {
    REID_Release(api);
  }
}

bool reid_feature_extract(
  AlgoHandle handle,
  BufferInfo * buffer,
  FeaturesType & features)
{
  float * feats = NULL;
  REIDHandle api = reinterpret_cast<REIDHandle>(handle);

  if (NULL == api || NULL == buffer || NULL == buffer->data) {
    return false;
  }

  XMReIDImage input;
  input.data = (unsigned char *)buffer->data;
  input.width = buffer->width;
  input.height = buffer->height;

  switch (buffer->format) {
    case FORMAT_BGR:
      input.fmt = XM_IMG_FMT_BGR;
      break;
    case FORMAT_RGB:
      input.fmt = XM_IMG_FMT_RGB;
      break;
    default:
      return false;
  }

  REID_ExtractFeat(api, &input, feats);

  if (feats != NULL) {
    memcpy(features.data(), feats, sizeof(float) * REID_GetFeatLen());
    delete feats;
  }

  return true;
}

int reid_get_feature_size()
{
  return REID_GetFeatLen();
}

double reid_similarity_one2one(
  AlgoHandle handle,
  FeaturesType & feats,
  FeaturesType & probe)
{
  REIDHandle api = reinterpret_cast<REIDHandle>(handle);

  return REID_GetSimOfOne2One(api, feats.data(), probe.data());
}

double reid_similarity_one2group(
  AlgoHandle handle,
  FeaturesType & feats,
  FeaturesType & probe,
  int num_probe, GroupMatchMethod method)
{
  REIDHandle api = reinterpret_cast<REIDHandle>(handle);

  return REID_GetSimOfOne2Group(api, feats.data(), probe.data(), num_probe, method);
}

double reid_similarity_group2group(
  AlgoHandle handle,
  FeaturesType & feats, int num_feats,
  FeaturesType & probe, int num_probe,
  GroupMatchMethod method)
{
  REIDHandle api = reinterpret_cast<REIDHandle>(handle);

  return REID_GetSimOfGroup2Group(api, feats.data(), num_feats, probe.data(), num_probe, method);
}

bool reid_match_by_similarity(double similarity)
{
  return REID_JudgeIfSame(similarity);
}
