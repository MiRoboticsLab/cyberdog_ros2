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

#define LOG_TAG "MonoStream"
#include <string>
#include <memory>
#include <utility>
#include "stereo_camera/mono_stream_consumer.hpp"
#include "camera_utils/log.hpp"

namespace cyberdog_camera
{

MonoStreamConsumer::MonoStreamConsumer(Size2D<uint32_t> size, const std::string & name)
: StreamConsumer(size),
  m_name(name)
{
  rclcpp::Node::SharedPtr node = std::make_shared<rclcpp::Node>("camera_" + m_name);
  auto qos = rclcpp::QoS(rclcpp::QoSInitialization(RMW_QOS_POLICY_HISTORY_KEEP_LAST, 10));
  qos.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
  m_publisher = node->create_publisher<sensor_msgs::msg::Image>("image_" + m_name, qos);
}

MonoStreamConsumer::~MonoStreamConsumer()
{
}

bool MonoStreamConsumer::threadInitialize()
{
  if (!StreamConsumer::threadInitialize()) {
    return false;
  }

  return true;
}

bool MonoStreamConsumer::threadShutdown()
{
  return StreamConsumer::threadShutdown();
}

bool MonoStreamConsumer::processBuffer(Buffer * buffer)
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

  ImageBuffer buf;
  memset(&buf, 0, sizeof(buf));
  buf.res = m_size;
  buf.fd = fd;
  buf.timestamp = ts;

  publishImage(buf);
  m_frameCount++;
  bufferDone(buffer);

  return true;
}

void MonoStreamConsumer::publishImage(ImageBuffer & buf)
{
  auto msg = std::make_unique<sensor_msgs::msg::Image>();

  msg->is_bigendian = false;
  msg->width = buf.res.width();
  msg->height = buf.res.height();
  msg->encoding = "mono8";
  msg->step = static_cast<sensor_msgs::msg::Image::_step_type>(msg->width);
  size_t size = msg->step * msg->height;
  msg->data.resize(size);
  NvBuffer2Raw(buf.fd, 0, msg->width, msg->height, &msg->data[0]);
  msg->header.frame_id = m_name + "_link";
  msg->header.stamp.sec = buf.timestamp.tv_sec;
  msg->header.stamp.nanosec = buf.timestamp.tv_nsec;

  m_publisher->publish(std::move(msg));
}

}  // namespace cyberdog_camera
