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

#define LOG_TAG "StreamConsumer"
#include "camera_base/stream_consumer.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

StreamConsumer::StreamConsumer(Size2D<uint32_t> size, NvBufferLayout layout, int num_buf)
: m_size(size),
  m_layout(layout),
  m_maxBuffersNum(num_buf),
  m_frameCount(0),
  m_firstFrameTime(0),
  m_fps(0),
  m_eglDisplay(EGL_NO_DISPLAY)
{
  if (m_maxBuffersNum > MAX_BUFFERS_NUM) {
    m_maxBuffersNum = MAX_BUFFERS_NUM;
  }
}

StreamConsumer::~StreamConsumer()
{
}

bool StreamConsumer::threadInitialize()
{
  CAM_INFO("m_maxBuffersNum = %d", m_maxBuffersNum);

  Argus::IBufferOutputStream * iBufferOutputStream =
    Argus::interface_cast<Argus::IBufferOutputStream>(m_outputStream.get());
  if (!iBufferOutputStream) {
    CAM_ERR("Failed to get IBufferOutputStream interface.");
    return false;
  }

  for (int i = 0; i < m_maxBuffersNum; i++) {
    m_nativeBuffers[i] =
      cyberdog_camera::DmaBuffer::create(m_size, NvBufferColorFormat_NV12, m_layout);
    if (!m_nativeBuffers[i]) {
      CAM_ERR("Failed to create %dth native buffer.", i);
      return false;
    }
  }

  m_eglDisplay = eglGetDisplay(EGL_DEFAULT_DISPLAY);
  if (m_eglDisplay == EGL_NO_DISPLAY) {
    CAM_ERR("Failed get EGL display.");
    return false;
  }
  for (int i = 0; i < m_maxBuffersNum; i++) {
    m_eglImages[i] = m_nativeBuffers[i]->createEGLImage(m_eglDisplay);
    if (m_eglImages[i] == EGL_NO_IMAGE_KHR) {
      CAM_ERR("Failed to create %dth egl image.", i);
      return false;
    }
  }

  /* Create the BufferSettings object to configure Buffer creation */
  Argus::UniqueObj<Argus::BufferSettings> bufferSettings(
    iBufferOutputStream->createBufferSettings());
  Argus::IEGLImageBufferSettings * iBufferSettings =
    Argus::interface_cast<Argus::IEGLImageBufferSettings>(bufferSettings.get());
  if (!iBufferSettings) {
    CAM_ERR("Failed to create BufferSettings");
    return false;
  }

  for (int i = 0; i < m_maxBuffersNum; i++) {
    iBufferSettings->setEGLImage(m_eglImages[i]);
    iBufferSettings->setEGLDisplay(m_eglDisplay);
    m_buffers[i].reset(iBufferOutputStream->createBuffer(bufferSettings.get()));
    IBuffer * iBuffer = Argus::interface_cast<IBuffer>(m_buffers[i].get());

    iBuffer->setClientData(m_nativeBuffers[i]);
    m_nativeBuffers[i]->setArgusBuffer(m_buffers[i].get());
    m_bufferMap[m_nativeBuffers[i]->getFd()] = m_buffers[i].get();

    if (!Argus::interface_cast<Argus::IEGLImageBuffer>(m_buffers[i].get())) {
      CAM_ERR("Failed to create Buffer");
      return false;
    }
    if (iBufferOutputStream->releaseBuffer(m_buffers[i].get()) != Argus::STATUS_OK) {
      CAM_ERR("Failed to release Buffer for capture use");
      return false;
    }
  }

  return true;
}

bool StreamConsumer::threadExecute()
{
  Argus::IBufferOutputStream * iBufferOutputStream =
    Argus::interface_cast<Argus::IBufferOutputStream>(m_outputStream.get());
  if (!iBufferOutputStream) {
    CAM_ERR("Failed to get IBufferOutputStream interface");
    return false;
  }

  while (true) {
    Argus::Status status = Argus::STATUS_OK;
    Buffer * buffer = iBufferOutputStream->acquireBuffer(Argus::TIMEOUT_INFINITE, &status);
    if (!processBuffer(buffer)) {
      CAM_INFO("stop stream thread");
      /* Timeout or error happen, exit */
      break;
    }
  }

  requestShutdown();

  return true;
}

bool StreamConsumer::endOfStream()
{
  Argus::IBufferOutputStream * iBufferOutputStream =
    Argus::interface_cast<Argus::IBufferOutputStream>(m_outputStream.get());
  if (!iBufferOutputStream) {
    CAM_INFO("Failed to get IEGLOutputStream interface");
    return false;
  }

  // end the buffer stream
  iBufferOutputStream->endOfStream();

  return true;
}

bool StreamConsumer::threadShutdown()
{
  m_outputStream.reset();
  /* Destroy the EGLImages */
  for (int i = 0; i < m_maxBuffersNum; i++) {
    NvDestroyEGLImage(NULL, m_eglImages[i]);
  }

  /* Destroy the native buffers */
  for (int i = 0; i < m_maxBuffersNum; i++) {
    delete m_nativeBuffers[i];
  }

  for (int i = 0; i < m_maxBuffersNum; i++) {
    m_buffers[i].reset();
  }

  m_bufferMap.clear();

  return true;
}

bool StreamConsumer::bufferDone(Buffer * buffer)
{
  Argus::IBufferOutputStream * iBufferOutputStream =
    Argus::interface_cast<Argus::IBufferOutputStream>(m_outputStream.get());
  if (!iBufferOutputStream) {
    CAM_ERR("Failed to get IBufferOutputStream interface");
    return false;
  }
  iBufferOutputStream->releaseBuffer(buffer);

  return true;
}

bool StreamConsumer::getFrameRate(float & frameRate)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);

  if (m_firstFrameTime == 0) {
    m_firstFrameTime = tv.tv_sec * 1000 * 1000 + tv.tv_usec;
  } else {
    uint64_t current = tv.tv_sec * 1000 * 1000 + tv.tv_usec;

    m_fps++;
    if (current - m_firstFrameTime > 1 * 1000 * 1000) {
      frameRate = static_cast<float>(m_fps) *
        (1e6 / static_cast<double>(current - m_firstFrameTime));
      m_firstFrameTime = current;
      m_fps = 0;

      return true;
    }
  }

  return false;
}

}  // namespace cyberdog_camera
