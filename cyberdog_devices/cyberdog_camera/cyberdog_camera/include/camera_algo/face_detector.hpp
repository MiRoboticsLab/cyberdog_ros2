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

#ifndef CAMERA_ALGO__FACE_DETECTOR_HPP_
#define CAMERA_ALGO__FACE_DETECTOR_HPP_

#include <vector>
#include "camera_base/camera_buffer.hpp"

#ifdef HAVE_VISION_LIB
#include <cyberdog_vision/face_detect_api.h>
#endif

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
class FaceDetector
{
public:
  static FaceDetector * create();
  void match(ImageBuffer & image, FeatureMap & feats, std::vector<MatchedFaceInfo> & out);
  void detect(ImageBuffer & image, std::vector<SingleFaceInfo> & out);

  ~FaceDetector();

private:
  FaceDetector();
  bool initialize();

  bool m_init;
  AlgoHandle m_apiHandle;
};
#endif

}  // namespace cyberdog_camera

#endif  // CAMERA_ALGO__FACE_DETECTOR_HPP_
