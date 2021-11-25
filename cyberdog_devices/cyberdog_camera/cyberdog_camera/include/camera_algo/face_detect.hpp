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

#ifndef CAMERA_ALGO__FACE_DETECT_HPP_
#define CAMERA_ALGO__FACE_DETECT_HPP_

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include <memory>
#include "camera_service/ros2_service.hpp"
#include "./algorithm_base.hpp"
#include "camera_service/face_manager.hpp"
#include "./face_detector.hpp"
#include "camera_base/jpeg_encoder.hpp"

namespace cyberdog_camera
{

#ifdef HAVE_VISION_LIB
class FaceDetect : public AlgorithmBase
{
public:
  FaceDetect();
  virtual ~FaceDetect();

  virtual bool init();
  virtual bool processImageBuffer(ImageBuffer & buffer);
  virtual bool isIdle()
  {
    return !m_inProcess;
  }

  int addFace(const std::string & name, bool is_host);
  int cancelAddFace();

private:
  enum FaceStatsType
  {
    statsFaceNum = 0,
    statsFaceYaw,
    statsFacePitch,
    statsFaceRow,
    statsFaceArea,
    statsFaceTypeMax,
  };

  enum MassageType
  {
    msgNumError = 0,
    msgPoseError,
    msgDistError,
    msgTypeMax,
  };

  void processFaceResult(ImageBuffer & frame, std::vector<MatchedFaceInfo> & faces);
  bool processFaceInfo(std::vector<SingleFaceInfo> & infos);
  bool checkFacePose(SingleFaceInfo & info);
  bool cropFaceRegion(int fd, const cv::Rect & rect, unsigned char ** jpeg_buf, size_t & size);
  cv::Rect getFaceImageRect(Rect & rect);
  std::vector<SingleFaceInfo> filter_face_info(std::vector<SingleFaceInfo> & infos);
  void clearFaceStats();

  FaceDetector * m_detector;
  rclcpp::Publisher<FaceInfoT>::SharedPtr pub_;
  bool m_addFace;
  FaceId m_faceId;
  std::shared_ptr<cyberdog_camera::JpegEncoder> m_jpegEncoder;
  std::atomic<bool> m_inProcess;

  SizedVector<float, 10> m_faceStats[statsFaceTypeMax];
  unsigned * m_msgSendFlags;
};
#else
class FaceDetect : public AlgorithmBase
{
public:
  FaceDetect()
  {
  }
  virtual ~FaceDetect()
  {
  }

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

  int addFace(const std::string & name, bool is_host) {return 0;}
  int cancelAddFace() {return 0;}
};
#endif

}  // namespace cyberdog_camera

#endif  // CAMERA_ALGO__FACE_DETECT_HPP_
