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

#ifndef CAMERA_ALGO__FEATURE_DETECTOR_HPP_
#define CAMERA_ALGO__FEATURE_DETECTOR_HPP_

#include <vector>
#include "camera_base/camera_buffer.hpp"

#ifdef HAVE_VISION_LIB
#include <cyberdog_vision/reid_api.h>
#endif

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
class FeatureDetector
{
public:
  static FeatureDetector * create();
  void destroy();

  bool extract(ImageBuffer & image, std::vector<float> & feats);
  bool extract(ImageBuffer & image, float * & feats);
  double getMatchScore(std::vector<float> & probe, std::vector<float> & feats);
  double getMatchScore(
    std::vector<float> & probe, std::vector<float> & feats,
    int probeNum);
  bool isMatched(double score);
  int getFeatureSize();

private:
  FeatureDetector();
  ~FeatureDetector();

  bool initialize();
  AlgoHandle m_apiHandle;
};
#endif

}  // namespace cyberdog_camera

#endif  // CAMERA_ALGO__FEATURE_DETECTOR_HPP_
