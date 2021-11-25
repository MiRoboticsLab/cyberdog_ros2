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

#ifndef CAMERA_BASE__VIDEO_MUXER_HPP_
#define CAMERA_BASE__VIDEO_MUXER_HPP_

extern "C"
{
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
};

namespace cyberdog_camera
{

class VideoMuxer
{
public:
  static VideoMuxer * create(int width, int height, const char * filename);
  ~VideoMuxer();

  bool initialize(int width, int height, const char * filename);
  void release();
  bool processData(uint8_t * data, size_t size, int64_t timestamp);

private:
  VideoMuxer();

  AVFormatContext * m_avContext;
  AVStream * m_avStream;
  uint64_t m_frameCount;
  int64_t m_firstTs;
  int64_t m_lastTs;
};

}  // namespace cyberdog_camera
#endif  // CAMERA_BASE__VIDEO_MUXER_HPP_
