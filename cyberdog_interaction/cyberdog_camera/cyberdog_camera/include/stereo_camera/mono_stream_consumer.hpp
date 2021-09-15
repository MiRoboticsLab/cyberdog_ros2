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

#ifndef STEREO_CAMERA__MONO_STREAM_CONSUMER_HPP_
#define STEREO_CAMERA__MONO_STREAM_CONSUMER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <string>
#include "camera_base/stream_consumer.hpp"

namespace cyberdog_camera
{

class MonoStreamConsumer : public StreamConsumer
{
public:
  MonoStreamConsumer(Size2D<uint32_t> size, const std::string & name);
  virtual ~MonoStreamConsumer();

  virtual bool threadInitialize();
  virtual bool threadShutdown();
  virtual bool processBuffer(Buffer * buffer);

private:
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_publisher;

  void publishImage(ImageBuffer & buf);
  const std::string m_name;
};

}  // namespace cyberdog_camera

#endif  // STEREO_CAMERA__MONO_STREAM_CONSUMER_HPP_
