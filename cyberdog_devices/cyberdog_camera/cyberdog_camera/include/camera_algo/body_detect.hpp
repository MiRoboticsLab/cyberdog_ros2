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

#ifndef CAMERA_ALGO__BODY_DETECT_HPP_
#define CAMERA_ALGO__BODY_DETECT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <vector>
#include <map>
#include "camera_service/ros2_service.hpp"
#include "./algorithm_base.hpp"
#include "./body_detector.hpp"
#include "./feature_detector.hpp"

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
class BodyDetect : public AlgorithmBase
{
public:
  BodyDetect();
  virtual ~BodyDetect();

  //  Override
  virtual bool init();
  virtual bool processImageBuffer(ImageBuffer & buffer);
  virtual bool isIdle()
  {
    return !m_inProcess;
  }

  void setReidEnabled(bool enable)
  {
    m_reidEnabled = enable;
  }

  void setReidObject(std::vector<int> bbox)
  {
    m_reidObject.rect.left = bbox[0];
    m_reidObject.rect.top = bbox[1];
    m_reidObject.rect.width = bbox[2];
    m_reidObject.rect.height = bbox[3];
    m_reidObject.feats.clear();
    m_id++;
  }

  bool isBodyTracked() {return m_isTracked;}

private:
  void processBodyResult(ImageBuffer & frame, std::vector<SingleBodyInfo> & bodys);
  bool getBodyFeatures(ImageBuffer & frame, std::vector<SingleBodyInfo> & bodys);

  BodyDetector * m_bodyDetector;
  FeatureDetector * m_featDetector;
  rclcpp::Publisher<BodyInfoT>::SharedPtr pub_;
  std::map<int, std::vector<float>> m_reidMap;
  SingleBodyInfo m_reidObject;
  bool m_reidEnabled;
  int m_unMatchCount;
  int m_id;
  bool m_isTracked;
  std::atomic<bool> m_inProcess;
};
#else
class BodyDetect : public AlgorithmBase
{
public:
  BodyDetect()
  {
  }
  virtual ~BodyDetect()
  {
  }

  //  Override
  virtual bool init() {return true;}
  virtual bool processImageBuffer(ImageBuffer & buffer)
  {
    if (buffer.data == NULL) {
      return false;
    }

    bufferDone(buffer);
    return true;
  }
  virtual bool isIdle() {return true;}

  void setReidEnabled(bool enable) {}

  void setReidObject(std::vector<int> bbox) {}

  bool isBodyTracked() {return false;}
};
#endif

}  // namespace cyberdog_camera

#endif  // CAMERA_ALGO__BODY_DETECT_HPP_
