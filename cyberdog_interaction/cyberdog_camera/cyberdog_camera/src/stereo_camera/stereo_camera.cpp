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

#include <string>
#include <memory>
#include "stereo_camera/stereo_camera.hpp"
#include "stereo_camera/mono_stream_consumer.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

bool CameraHolder::initialize(int camera_id, const std::string & name)
{
  if (m_captureSession) {
    return true;
  }

  if (!CameraDispatcher::getInstance().createSession(m_captureSession, camera_id)) {
    CAM_ERR("Failed to create CaptureSession");
    return false;
  }

  m_stream = new MonoStreamConsumer(Size2D<uint32_t>(640, 480), name);
  if (m_stream == NULL) {
    CAM_ERR("Failed to create stream.");
    return false;
  }

  UniqueObj<OutputStream> outputStream;
  if (!CameraDispatcher::getInstance().createOutputStream(
      m_captureSession.get(), outputStream, m_stream->getSize()))
  {
    CAM_ERR("Failed to create output stream");
    return false;
  }

  m_stream->setOutputStream(outputStream.release());
  m_stream->initialize();
  m_stream->waitRunning();

  if (!CameraDispatcher::getInstance().createRequest(
      m_captureSession.get(), m_previewRequest, Argus::CAPTURE_INTENT_PREVIEW))
  {
    CAM_ERR("Failed to create preview Request");
    return false;
  }

  CameraDispatcher::getInstance().enableOutputStream(
    m_previewRequest.get(), m_stream->getOutputStream());

  CAM_INFO("Starting repeat capture requests.\n");
  if (!CameraDispatcher::getInstance().startRepeat(
      m_captureSession.get(),
      m_previewRequest.get()))
  {
    CAM_ERR("Failed to start repeat capture request");
    return false;
  }

  return true;
}

bool CameraHolder::shutdown()
{
  CameraDispatcher::getInstance().stopRepeat(m_captureSession.get());
  CameraDispatcher::getInstance().waitForIdle(m_captureSession.get());
  CAM_INFO("Stop repeat capture requests.\n");

  CameraDispatcher::getInstance().disableOutputStream(
    m_previewRequest.get(), m_stream->getOutputStream());

  m_stream->endOfStream();
  m_stream->shutdown();
  delete m_stream;

  m_previewRequest.reset();
  m_captureSession.reset();

  return true;
}

const int StereoCameraNode::g_cameraIds[] = {1, 2};
const char * StereoCameraNode::g_cameraStrings[] = {"left", "right"};

StereoCameraNode::StereoCameraNode(const std::string & name)
: rclcpp::Node(name)
{
  m_cameras.resize(2);
  initialize();
}

StereoCameraNode::~StereoCameraNode()
{
  shutdown();
}

bool StereoCameraNode::initialize()
{
  for (size_t i = 0; i < m_cameras.size(); i++) {
    m_cameras[i] = new CameraHolder();
    if (!m_cameras[i]->initialize(g_cameraIds[i], g_cameraStrings[i])) {
      CAM_ERR("Failed to initialize camera %d", g_cameraIds[i]);
    }
  }

  return true;
}

bool StereoCameraNode::shutdown()
{
  for (size_t i = 0; i < m_cameras.size(); i++) {
    if (!m_cameras[i]->shutdown()) {
      CAM_ERR("Failed to shutdown camera %d", g_cameraIds[i]);
    }

    delete m_cameras[i];
    m_cameras[i] = NULL;
  }

  return true;
}

}  // namespace cyberdog_camera

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<cyberdog_camera::StereoCameraNode>("stereo_camera"));
  rclcpp::shutdown();

  return 0;
}
