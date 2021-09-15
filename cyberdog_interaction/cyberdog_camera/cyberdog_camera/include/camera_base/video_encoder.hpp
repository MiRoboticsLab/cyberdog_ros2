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

#ifndef CAMERA_BASE__VIDEO_ENCODER_HPP_
#define CAMERA_BASE__VIDEO_ENCODER_HPP_

#include <set>
#include <string>
#include <NvVideoEncoder.h>

class NvBuffer;

namespace cyberdog_camera
{

struct video_encoder_settings
{
  uint32_t bitrate;
  int profile;
  bool insert_sps;
};

class VideoEncoder
{
public:
  VideoEncoder(
    const char * name, int width,
    int height, uint32_t pixfmt = V4L2_PIX_FMT_H265);
  ~VideoEncoder();

  bool initialize(video_encoder_settings & settings);
  bool shutdown();

  bool encodeFromFd(int dmabuf_fd);

  // Callbackt to return buffer
  void setInputDoneCallback(void (* callback)(int, void *), void * arg)
  {
    m_inputDonecb = callback;
    m_inputCbArg = arg;
  }

  // Callback to process output data
  void setOutputDoneCallback(void (* callback)(uint8_t *, size_t, int64_t, void *), void * arg)
  {
    m_outputDoneCb = callback;
    m_outputCbArg = arg;
  }

private:
  NvVideoEncoder * m_VideoEncoder;
  bool createVideoEncoder(video_encoder_settings & settings);

  static bool encoderCapturePlaneDqCallback(
    struct v4l2_buffer * v4l2_buf,
    NvBuffer * buffer,
    NvBuffer * shared_buffer,
    void * arg)
  {
    VideoEncoder * thiz = static_cast<VideoEncoder *>(arg);
    return thiz->encoderCapturePlaneDqCallback(v4l2_buf, buffer, shared_buffer);
  }

  bool encoderCapturePlaneDqCallback(
    struct v4l2_buffer * v4l2_buf,
    NvBuffer * buffer,
    NvBuffer * shared_buffer);

  std::string m_name;
  int m_width;
  int m_height;
  uint32_t m_pixfmt;

  std::set<int> m_dmabufFdSet;         // Collection to track all queued buffer
  void (* m_inputDonecb)(int, void *);          // Output plane DQ callback
  void (* m_outputDoneCb)(uint8_t *, size_t, int64_t, void *);
  void * m_inputCbArg;
  void * m_outputCbArg;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_BASE__VIDEO_ENCODER_HPP_
