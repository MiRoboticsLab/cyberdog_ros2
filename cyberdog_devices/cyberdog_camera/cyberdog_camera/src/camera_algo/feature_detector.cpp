// Copyright (c) 2021  Beijing Xiaomi Mobile Software Co., Ltd.
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

#include <stdio.h>
#include <iostream>
#include <vector>
#include "camera_algo/feature_detector.hpp"

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
FeatureDetector * FeatureDetector::create()
{
  FeatureDetector * instance;

  instance = new FeatureDetector();
  instance->initialize();

  return instance;
}

FeatureDetector::FeatureDetector()
{
  m_apiHandle = NULL;
  printf("%s\n", __FUNCTION__);
}

FeatureDetector::~FeatureDetector()
{
  printf("%s\n", __FUNCTION__);
  if (m_apiHandle) {
    reid_destroy(m_apiHandle);
  }
}

void FeatureDetector::destroy()
{
  delete this;
}

bool FeatureDetector::initialize()
{
  m_apiHandle = reid_init();

  return true;
}

bool FeatureDetector::extract(ImageBuffer & image, std::vector<float> & feats)
{
  BufferInfo img;

  memset(&img, 0, sizeof(img));
  img.data = image.data;
  img.width = image.res.width();
  img.height = image.res.height();
  img.format = FORMAT_BGR;

  return reid_feature_extract(m_apiHandle, &img, feats);
}

double FeatureDetector::getMatchScore(std::vector<float> & probe, std::vector<float> & feats)
{
  return reid_similarity_one2one(m_apiHandle, probe, feats);
}

double FeatureDetector::getMatchScore(
  std::vector<float> & probe, std::vector<float> & feats,
  int probeNum)
{
  return reid_similarity_one2group(m_apiHandle, feats, probe, probeNum);
}

bool FeatureDetector::isMatched(double score)
{
  return reid_match_by_similarity(score);
}

int FeatureDetector::getFeatureSize()
{
  return reid_get_feature_size();
}
#endif

}  // namespace cyberdog_camera
