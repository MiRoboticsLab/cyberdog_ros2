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

#define LOG_TAG "CameraContext"

#include <string>
#include <memory>
#include <vector>
#include "camera_service/argus_camera_context.hpp"
#include "camera_base/camera_dispatcher.hpp"
#include "camera_service/video_stream_consumer.hpp"
#include "camera_service/h264_stream_consumer.hpp"
#include "camera_service/rgb_stream_consumer.hpp"
#include "camera_service/algo_stream_consumer.hpp"
#include "camera_service/ncs_client.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

const uint64_t ONE_SECONDS_IN_NANOSECONDS = 1000000000;
const int VIDEO_WIDTH_DEFAULT = 1280;
const int VIDEO_HEIGHT_DEFAULT = 960;
const int ALGO_WIDTH_DEFAULT = 1280;
const int ALGO_HEIGHT_DEFAULT = 960;
const int IMAGE_WIDTH_DEFAULT = 640;
const int IMAGE_HEIGHT_DEFAULT = 480;

ArgusCameraContext::ArgusCameraContext(int camId)
: m_cameraId(camId),
  m_isStreaming(false)
{
  initialize();
}

ArgusCameraContext::~ArgusCameraContext()
{
  shutdown();
}

bool ArgusCameraContext::initialize()
{
  CAM_INFO("%s", __FUNCTION__);

  for (int i = 0; i < STREAM_TYPE_MAX; i++) {
    m_activeStreams.push_back(NULL);
  }

  return true;
}

bool ArgusCameraContext::shutdown()
{
  CAM_INFO("%s", __FUNCTION__);

  if (m_captureSession) {
    closeSession();
  }

  return true;
}

bool ArgusCameraContext::createSession()
{
  CAM_INFO("%s", __FUNCTION__);

  return true;
}

bool ArgusCameraContext::closeSession()
{
  CAM_INFO("%s", __FUNCTION__);

  if (!m_captureSession) {
    return true;
  }

  deinitCameraContext();

  return true;
}

bool ArgusCameraContext::initCameraContext()
{
  return true;
}

bool ArgusCameraContext::deinitCameraContext()
{
  if (CameraDispatcher::getInstance().isRepeating(m_captureSession.get())) {
    CameraDispatcher::getInstance().stopRepeat(m_captureSession.get());
    CameraDispatcher::getInstance().waitForIdle(m_captureSession.get());
  }

  // Destroy the output stream to end the consumer thread,
  // and wait for the consumer thread to complete.
  for (unsigned i = 0; i < m_activeStreams.size(); i++) {
    if (m_activeStreams[i]) {
      m_activeStreams[i]->endOfStream();
      m_activeStreams[i]->shutdown();
      m_activeStreams[i] = NULL;
    }
  }

  NCSClient::getInstance().requestLed(false);
  m_previewRequest.reset();
  m_captureSession.reset();

  return true;
}

int ArgusCameraContext::takePicture(const char * path, int width, int height)
{
  if (!m_isStreaming) {
    CAM_ERR("Device is not stream on.");
    return CAM_INVALID_STATE;
  }

  UniqueObj<Argus::Request> captureRequest;
  if (!CameraDispatcher::getInstance().createRequest(
      m_captureSession.get(), captureRequest, Argus::CAPTURE_INTENT_STILL_CAPTURE))
  {
    CAM_ERR("Failed to create capture Request");
    return CAM_ERROR;
  }

  Size2D<uint32_t> captureSize;
  if (width == 0 || height == 0) {
    captureSize = getSensorSize();
  } else {
    captureSize = Size2D<uint32_t>(width, height);
  }

  UniqueObj<OutputStream> captureStream;
  if (!CameraDispatcher::getInstance().createOutputStream(
      m_captureSession.get(), captureStream, captureSize, Argus::STREAM_TYPE_EGL))
  {
    CAM_ERR("Failed to create capture output stream");
    return false;
  }

  if (!CameraDispatcher::getInstance().enableOutputStream(
      captureRequest.get(), captureStream.get()))
  {
    CAM_ERR("Failed to enable capture stream");
    return CAM_ERROR;
  }

  UniqueObj<EGLStream::FrameConsumer> consumer(
    EGLStream::FrameConsumer::create(captureStream.get()));
  EGLStream::IFrameConsumer * iFrameConsumer =
    Argus::interface_cast<EGLStream::IFrameConsumer>(consumer);
  if (!iFrameConsumer) {
    CAM_ERR("Failed to create FrameConsumer");
    return CAM_ERROR;
  }

  if (!CameraDispatcher::getInstance().capture(
      m_captureSession.get(), captureRequest.get()))
  {
    CAM_ERR("Failed to submit the still capture request");
    return CAM_ERROR;
  }

  UniqueObj<EGLStream::Frame> frame(iFrameConsumer->acquireFrame(ONE_SECONDS_IN_NANOSECONDS));
  if (!frame) {
    CAM_ERR("Failed to aquire frame");
    return CAM_ERROR;
  }

  // Use the IFrame interface to provide access to the Image in the Frame.
  EGLStream::IFrame * iFrame = Argus::interface_cast<EGLStream::IFrame>(frame);
  if (!iFrame) {
    CAM_ERR("Failed to get IFrame interface.");
    return CAM_ERROR;
  }

  EGLStream::Image * image = iFrame->getImage();
  if (!image) {
    CAM_ERR("Failed to get image.");
    return CAM_ERROR;
  }

  EGLStream::IImageJPEG * iJPEG = Argus::interface_cast<EGLStream::IImageJPEG>(image);
  if (!iJPEG) {
    CAM_ERR("Failed to get IImageJPEG interface.");
    return CAM_ERROR;
  }

  if (iJPEG->writeJPEG(path) != Argus::STATUS_OK) {
    CAM_ERR("Failed to write JPEG to '%s'", path);
    return CAM_ERROR;
  }
  NCSClient::getInstance().play(SoundShutter);

  return CAM_SUCCESS;
}

int ArgusCameraContext::startCameraStream(
  Streamtype type,
  Size2D<uint32_t> size, void * args)
{
  std::lock_guard<std::mutex> lock(m_streamLock);
  std::string * filename;
  std::shared_ptr<StreamConsumer> stream;

  if (m_activeStreams[type]) {
    return CAM_INVALID_STATE;
  }

  if (!m_captureSession) {
    if (!CameraDispatcher::getInstance().createSession(m_captureSession, m_cameraId)) {
      CAM_ERR("Failed to create CaptureSession");
      return false;
    }
  }

  if (!m_previewRequest) {
    if (!CameraDispatcher::getInstance().createRequest(
        m_captureSession.get(), m_previewRequest, Argus::CAPTURE_INTENT_PREVIEW))
    {
      CAM_ERR("Failed to create preview Request");
      return false;
    }
  }

  if (CameraDispatcher::getInstance().isRepeating(m_captureSession.get())) {
    CameraDispatcher::getInstance().stopRepeat(m_captureSession.get());
    CameraDispatcher::getInstance().waitForIdle(m_captureSession.get());
  }

  switch (type) {
    case STREAM_TYPE_ALGO:
      stream = std::make_shared<AlgoStreamConsumer>(size);
      break;
    case STREAM_TYPE_RGB:
      stream = std::make_shared<RGBStreamConsumer>(size);
      break;
    case STREAM_TYPE_VIDEO:
      filename = static_cast<std::string *>(args);
      CAM_INFO("filename = %s", filename->c_str());
      stream = std::make_shared<VideoStreamConsumer>(size, *filename);
      break;
    case STREAM_TYPE_H264:
      stream = std::make_shared<H264StreamConsumer>(size);
      break;
    default:
      return CAM_INVALID_ARG;
  }

  UniqueObj<OutputStream> outputStream;
  if (!CameraDispatcher::getInstance().createOutputStream(
      m_captureSession.get(), outputStream, stream->getSize()))
  {
    CAM_ERR("Failed to create output stream %d", type);
    return CAM_ERROR;
  }
  stream->setOutputStream(outputStream.release());
  stream->initialize();
  stream->waitRunning();

  m_activeStreams[type] = stream;
  CameraDispatcher::getInstance().enableOutputStream(
    m_previewRequest.get(), m_activeStreams[type]->getOutputStream());

  std::vector<OutputStream *> streams;
  CameraDispatcher::getInstance().getOutputStreams(m_previewRequest.get(), &streams);
  if (streams.size() > 0) {
    // Submit capture requests.
    CAM_INFO("Starting repeat capture requests.\n");
    if (!CameraDispatcher::getInstance().startRepeat(
        m_captureSession.get(),
        m_previewRequest.get()))
    {
      CAM_ERR("Failed to start repeat capture request");
      return false;
    }
    m_isStreaming = true;
    NCSClient::getInstance().requestLed(true);
  }
  return CAM_SUCCESS;
}

int ArgusCameraContext::stopCameraStream(Streamtype type)
{
  std::lock_guard<std::mutex> lock(m_streamLock);
  if (!m_activeStreams[type]) {
    return CAM_SUCCESS;
  }

  if (CameraDispatcher::getInstance().isRepeating(m_captureSession.get())) {
    CameraDispatcher::getInstance().stopRepeat(m_captureSession.get());
    CameraDispatcher::getInstance().waitForIdle(m_captureSession.get());
  }
  CameraDispatcher::getInstance().disableOutputStream(
    m_previewRequest.get(), m_activeStreams[type]->getOutputStream());

  m_activeStreams[type]->endOfStream();
  m_activeStreams[type]->shutdown();
  m_activeStreams[type] = NULL;

  std::vector<OutputStream *> streams;
  CameraDispatcher::getInstance().getOutputStreams(m_previewRequest.get(), &streams);
  if (streams.size() > 0) {
    // Submit capture requests.
    CAM_INFO("Starting repeat capture requests.\n");
    if (!CameraDispatcher::getInstance().startRepeat(
        m_captureSession.get(),
        m_previewRequest.get()))
    {
      CAM_ERR("Failed to start repeat capture request");
      return false;
    }
  } else {
    m_isStreaming = false;
    NCSClient::getInstance().requestLed(false);
    m_previewRequest.reset();
    m_captureSession.reset();
  }

  return CAM_SUCCESS;
}

int ArgusCameraContext::startPreview(std::string & usage)
{
  int ret = CAM_SUCCESS;
  ret = startCameraStream(
    STREAM_TYPE_H264,
    Size2D<uint32_t>(VIDEO_WIDTH_DEFAULT, VIDEO_HEIGHT_DEFAULT), NULL);

  if (ret == CAM_SUCCESS && usage == "preview") {
    NCSClient::getInstance().play(SoundLiveStart);
  }
  if (ret == CAM_INVALID_STATE) {
    ret = CAM_SUCCESS;
  }

  return ret;
}

int ArgusCameraContext::stopPreview()
{
  return stopCameraStream(STREAM_TYPE_H264);
}

int ArgusCameraContext::startRecording(std::string & filename, int width, int height)
{
  int ret = CAM_SUCCESS;

  if (width == 0 || height == 0) {
    width = VIDEO_WIDTH_DEFAULT;
    height = VIDEO_HEIGHT_DEFAULT;
  }

  ret = startCameraStream(
    STREAM_TYPE_VIDEO,
    Size2D<uint32_t>(width, height), &filename);

  if (ret == CAM_SUCCESS) {
    NCSClient::getInstance().play(SoundRecordStart);
    m_videoFilename = filename;
  }

  if (ret == CAM_INVALID_STATE) {
    ret = CAM_SUCCESS;
  }

  return ret;
}

int ArgusCameraContext::stopRecording(std::string & filename)
{
  filename = m_videoFilename;

  return stopCameraStream(STREAM_TYPE_VIDEO);
}

int ArgusCameraContext::startRgbStream()
{
  startCameraStream(
    STREAM_TYPE_ALGO, Size2D<uint32_t>(
      ALGO_WIDTH_DEFAULT,
      ALGO_HEIGHT_DEFAULT), NULL);
  startCameraStream(
    STREAM_TYPE_RGB, Size2D<uint32_t>(
      IMAGE_WIDTH_DEFAULT,
      IMAGE_HEIGHT_DEFAULT), NULL);

  return CAM_SUCCESS;
}

int ArgusCameraContext::stopRgbStream()
{
  stopCameraStream(STREAM_TYPE_ALGO);
  stopCameraStream(STREAM_TYPE_RGB);

  return CAM_SUCCESS;
}

bool ArgusCameraContext::isRecording()
{
  return !!m_activeStreams[STREAM_TYPE_VIDEO];
}

uint64_t ArgusCameraContext::getRecordingTime()
{
  if (!m_activeStreams[STREAM_TYPE_VIDEO]) {
    return 0;
  }
  VideoStreamConsumer * video_stream =
    reinterpret_cast<VideoStreamConsumer *>(m_activeStreams[STREAM_TYPE_VIDEO].get());

  return video_stream->getRecordingTime();
}

Size2D<uint32_t> ArgusCameraContext::getSensorSize()
{
  return CameraDispatcher::getInstance().getSensorSize(m_cameraId);
}

}  // namespace cyberdog_camera
