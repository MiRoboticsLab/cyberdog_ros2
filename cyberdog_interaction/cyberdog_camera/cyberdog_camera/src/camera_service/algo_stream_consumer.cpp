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

#define LOG_TAG "AlgoStream"
#include "camera_service/algo_stream_consumer.hpp"
#include "camera_utils/log.hpp"
#include "camera_algo/algo_dispatcher.hpp"

namespace cyberdog_camera
{

AlgoStreamConsumer::AlgoStreamConsumer(Size2D<uint32_t> size)
: StreamConsumer(size)
{
}

AlgoStreamConsumer::~AlgoStreamConsumer()
{
  freeBuffers();
}

ImageBuffer AlgoStreamConsumer::getBuffer()
{
  std::lock_guard<std::mutex> lock(m_bufferMutex);
  ImageBuffer buffer;
  for (auto it = m_freeBuffers.begin(); it != m_freeBuffers.end(); it++) {
    buffer = *it;
    m_freeBuffers.erase(it);
    return buffer;
  }
  CAM_INFO("get a new buffer");
  size_t size = m_size.width() * m_size.height() * 3;
  buffer.data = malloc(size);

  return buffer;
}

void AlgoStreamConsumer::putBuffer(ImageBuffer buffer)
{
  std::lock_guard<std::mutex> lock(m_bufferMutex);
  m_freeBuffers.push_back(buffer);
}

void AlgoStreamConsumer::freeBuffers()
{
  std::lock_guard<std::mutex> lock(m_bufferMutex);
  for (auto it = m_freeBuffers.begin(); it != m_freeBuffers.end(); it++) {
    auto buffer = *it;
    CAM_INFO("free a buffer");
    free(buffer.data);
  }
  m_freeBuffers.clear();
}

bool AlgoStreamConsumer::threadInitialize()
{
  if (!StreamConsumer::threadInitialize()) {
    return false;
  }

  NvBufferCreateParams input_params;
  input_params.width = m_size.width();
  input_params.height = m_size.height();
  input_params.payloadType = NvBufferPayload_SurfArray;
  input_params.nvbuf_tag = NvBufferTag_NONE;
  input_params.layout = NvBufferLayout_Pitch;
  input_params.colorFormat = NvBufferColorFormat_ABGR32;

  if (NvBufferCreateEx(&m_rgbaFd, &input_params) < 0) {
    CAM_ERR("Failed to create NvBuffer.");
    return false;
  }

  m_convert = new ColorConvert(m_size.width(), m_size.height());
  m_convert->initialze(m_rgbaFd);

  AlgoDispatcher::getInstance().setBufferDoneCallback(bufferDoneCallback, this);

  return true;
}

bool AlgoStreamConsumer::threadShutdown()
{
  m_convert->release();
  delete m_convert;

  if (m_rgbaFd > 0) {
    NvBufferDestroy(m_rgbaFd);
  }

  AlgoDispatcher::getInstance().setAlgoEnabled(ALGO_FACE_DETECT, false);
  AlgoDispatcher::getInstance().setAlgoEnabled(ALGO_BODY_DETECT, false);

  return StreamConsumer::threadShutdown();
}

bool AlgoStreamConsumer::processBuffer(Buffer * buffer)
{
  struct timespec ts;

  clock_gettime(CLOCK_REALTIME, &ts);
  if (!buffer) {
    return false;
  }

  float frameRate;
  if (getFrameRate(frameRate)) {
    CAM_INFO("%.2f frames per second", frameRate);
  }

  cyberdog_camera::DmaBuffer * dma_buf = cyberdog_camera::DmaBuffer::fromArgusBuffer(buffer);
  int fd = dma_buf->getFd();
  ImageBuffer buf = getBuffer();
  buf.res = m_size;
  buf.timestamp = ts;
  buf.fd = fd;

  if (AlgoDispatcher::getInstance().needProcess(m_frameCount)) {
    NvBufferTransformParams transform_params;
    NvBufferRect src_rect, dest_rect;
    memset(&transform_params, 0, sizeof(transform_params));

    src_rect.top = 0;
    src_rect.left = 0;
    src_rect.width = m_size.width();
    src_rect.height = m_size.height();
    dest_rect.top = 0;
    dest_rect.left = 0;
    dest_rect.width = m_size.width();
    dest_rect.height = m_size.height();

    transform_params.transform_flag = NVBUFFER_TRANSFORM_FILTER;
    transform_params.transform_flip = NvBufferTransform_None;
    transform_params.transform_filter = NvBufferTransform_Filter_Nicest;
    transform_params.src_rect = src_rect;
    transform_params.dst_rect = dest_rect;

    NvBufferTransform(fd, m_rgbaFd, &transform_params);
    m_convert->convertRGBAToBGR(buf.data);

    AlgoDispatcher::getInstance().processImageBuffer(m_frameCount, buf);
  } else {
    imageBufferDone(buf);
  }

  m_frameCount++;

  return true;
}

void AlgoStreamConsumer::imageBufferDone(ImageBuffer buffer)
{
  int fd = buffer.fd;
  Argus::Buffer * dma_buffer = getBufferFromFd(fd);

  if (dma_buffer) {
    bufferDone(dma_buffer);
  }

  putBuffer(buffer);
}

}  // namespace cyberdog_camera
