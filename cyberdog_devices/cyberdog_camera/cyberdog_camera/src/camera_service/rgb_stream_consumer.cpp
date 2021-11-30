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

#define LOG_TAG "RGBStream"
#include <memory>
#include <utility>
#include "camera_service/rgb_stream_consumer.hpp"
#include "camera_service/camera_manager.hpp"
#include "camera_utils/log.hpp"
#include "camera_utils/utils.hpp"

namespace cyberdog_camera
{

RGBStreamConsumer::RGBStreamConsumer(Size2D<uint32_t> size)
: StreamConsumer(size),
  m_buffer(NULL)
{
  auto qos = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 10));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  m_publisher =
    CameraManager::getInstance()->create_publisher<sensor_msgs::msg::Image>("image", qos);
}

RGBStreamConsumer::~RGBStreamConsumer()
{
}

bool RGBStreamConsumer::threadInitialize()
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
    CAM_INFO("Failed to create NvBuffer.");
    return false;
  }

  // rgb buffer
  m_buffer = (unsigned char *)malloc(m_size.width() * m_size.height() * 3);
  if (!m_buffer) {
    CAM_INFO("Failed to malloc Buffer.");
    return false;
  }

  m_convert = new ColorConvert(m_size.width(), m_size.height());
  m_convert->initialze(m_rgbaFd);

  return true;
}

bool RGBStreamConsumer::threadShutdown()
{
  if (m_buffer) {
    free(m_buffer);
  }

  m_convert->release();
  delete m_convert;

  if (m_rgbaFd > 0) {
    NvBufferDestroy(m_rgbaFd);
  }

  return StreamConsumer::threadShutdown();
}

bool RGBStreamConsumer::processBuffer(Buffer * buffer)
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

  // Transform yuv to rgba
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

  // convert rgba to rgb
  m_convert->convertRGBAToBGR(m_buffer);

  ImageBuffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.res = m_size;
  buf.data = m_buffer;
  buf.timestamp = ts;

  publishImage(m_frameCount, buf);
  m_frameCount++;
  bufferDone(buffer);

  return true;
}

void RGBStreamConsumer::publishImage(uint64_t frame_id, ImageBuffer & buf)
{
  auto msg = std::make_unique<sensor_msgs::msg::Image>();

  msg->is_bigendian = false;
  msg->width = buf.res.width();
  msg->height = buf.res.height();
  msg->encoding = "bgr8";
  msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(msg->width * 3);
  size_t size = msg->step * msg->height;
  msg->data.resize(size);
  memcpy(&msg->data[0], buf.data, size);
  msg->header.frame_id = std::to_string(frame_id);
  msg->header.stamp.sec = buf.timestamp.tv_sec;
  msg->header.stamp.nanosec = buf.timestamp.tv_nsec;

  CAM_DEBUG("Publishing image #%zu", frame_id);
  m_publisher->publish(std::move(msg));
}

}  // namespace cyberdog_camera
