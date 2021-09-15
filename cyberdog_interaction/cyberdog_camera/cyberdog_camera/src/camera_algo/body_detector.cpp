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
#include <vector>
#include "camera_algo/body_detector.hpp"
#include "camera_utils/utils.hpp"

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
BodyDetector * BodyDetector::create()
{
  BodyDetector * instance;

  instance = new BodyDetector();
  instance->initialize();

  return instance;
}

BodyDetector::BodyDetector()
: m_init(false)
{
  m_apiHandle = NULL;
  printf("%s\n", __FUNCTION__);
}

BodyDetector::~BodyDetector()
{
  printf("%s\n", __FUNCTION__);

  if (NULL != m_apiHandle) {
    body_detect_destroy(m_apiHandle);
  }
}

bool BodyDetector::initialize()
{
  if (!m_init) {
    printf("BodyDetector::initialize()\n");

    m_apiHandle = body_detect_init();
    if (NULL != m_apiHandle) {
      m_init = true;
    }
  }

  return true;
}

void BodyDetector::detect(ImageBuffer & image, std::vector<SingleBodyInfo> & out)
{
  BufferInfo img;

  memset(&img, 0, sizeof(img));
  img.data = image.data;
  img.width = image.res.width();
  img.height = image.res.height();
  img.format = FORMAT_BGR;

  if (!body_detect(m_apiHandle, &img, out)) {
    CAM_ERR("Failed to detect bodies.");
  }
}
#endif

}  // namespace cyberdog_camera
