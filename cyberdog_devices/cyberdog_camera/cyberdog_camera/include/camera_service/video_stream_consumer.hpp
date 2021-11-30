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

#ifndef CAMERA_SERVICE__VIDEO_STREAM_CONSUMER_HPP_
#define CAMERA_SERVICE__VIDEO_STREAM_CONSUMER_HPP_

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <string>
#include "camera_base/stream_consumer.hpp"
#include "camera_base/video_muxer.hpp"
#include "camera_base/video_encoder.hpp"

namespace cyberdog_camera
{

class VideoStreamConsumer : public StreamConsumer
{
public:
  VideoStreamConsumer(Size2D<uint32_t> size, std::string & filename);
  virtual ~VideoStreamConsumer();

  virtual bool threadInitialize();
  virtual bool threadShutdown();
  virtual bool processBuffer(Buffer * buffer);
  uint64_t getRecordingTime();
  std::string & getFileName() {return m_filename;}

private:
  static void inputDoneCallback(int dmabuf_fd, void * arg)
  {
    VideoStreamConsumer * thiz = static_cast<VideoStreamConsumer *>(arg);
    thiz->inputDoneCallback(dmabuf_fd);
  }
  static void outputDoneCallback(uint8_t * data, size_t size, int64_t ts, void * arg)
  {
    VideoStreamConsumer * thiz = static_cast<VideoStreamConsumer *>(arg);
    thiz->outputDoneCallback(data, size, ts);
  }

  void inputDoneCallback(int fd);
  void outputDoneCallback(uint8_t * data, size_t size, int64_t ts);
  bool startSoundTimer();
  void stopSoundTimer();
  static void playVideoSound(union sigval val);

  VideoEncoder * m_videoEncoder;
  VideoMuxer * m_videoMuxer;
  std::string m_filename;
  uint64_t m_startTs;
  timer_t m_videoTimer;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_SERVICE__VIDEO_STREAM_CONSUMER_HPP_
