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

#ifndef STEREO_CAMERA__STEREO_CAMERA_HPP_
#define STEREO_CAMERA__STEREO_CAMERA_HPP_

#include <rclcpp/rclcpp.hpp>
#include <string>
#include <vector>
#include "camera_base/camera_dispatcher.hpp"
#include "camera_base/stream_consumer.hpp"

namespace cyberdog_camera
{

class CameraHolder
{
public:
  CameraHolder()
  {
  }
  ~CameraHolder()
  {
  }

  bool initialize(int camera_id, const std::string & name);
  bool shutdown();

private:
  Argus::UniqueObj<CaptureSession> m_captureSession;
  Argus::UniqueObj<Argus::Request> m_previewRequest;
  StreamConsumer * m_stream;
};

class StereoCameraNode : public rclcpp::Node
{
public:
  explicit StereoCameraNode(const std::string & name);
  ~StereoCameraNode();

private:
  bool initialize();
  bool shutdown();

  std::vector<CameraHolder *> m_cameras;
  static const int g_cameraIds[];
  static const char * g_cameraStrings[];
};

}  // namespace cyberdog_camera

#endif  // STEREO_CAMERA__STEREO_CAMERA_HPP_
