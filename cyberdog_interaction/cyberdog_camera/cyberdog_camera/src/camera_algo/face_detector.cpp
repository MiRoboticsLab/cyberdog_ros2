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
#include "camera_algo/face_detector.hpp"
#include "camera_utils/utils.hpp"

#define MATCH_THRE 0.65

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
FaceDetector * FaceDetector::create()
{
  FaceDetector * instance;

  instance = new FaceDetector();
  instance->initialize();

  return instance;
}

FaceDetector::FaceDetector()
: m_init(false)
{
  m_apiHandle = NULL;
}

FaceDetector::~FaceDetector()
{
  if (m_apiHandle) {
    face_detect_destroy(m_apiHandle);
  }
}

bool FaceDetector::initialize()
{
  if (!m_init) {
    m_apiHandle = face_detect_init(MATCH_THRE);
    if (NULL != m_apiHandle) {
      m_init = true;
    }
  }

  return true;
}

void FaceDetector::match(
  ImageBuffer & image, FeatureMap & feats,
  std::vector<MatchedFaceInfo> & out)
{
  BufferInfo img;

  memset(&img, 0, sizeof(img));
  img.data = image.data;
  img.width = image.res.width();
  img.height = image.res.height();
  img.format = FORMAT_BGR;

  face_match(m_apiHandle, &img, feats, out);
}

void FaceDetector::detect(ImageBuffer & image, std::vector<SingleFaceInfo> & out)
{
  BufferInfo img;

  memset(&img, 0, sizeof(img));
  img.data = image.data;
  img.width = image.res.width();
  img.height = image.res.height();
  img.format = FORMAT_BGR;

  face_detect(m_apiHandle, &img, out);
}
#endif

}  // namespace cyberdog_camera
