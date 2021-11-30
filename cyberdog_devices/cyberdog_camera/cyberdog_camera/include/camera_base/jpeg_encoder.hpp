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

#ifndef CAMERA_BASE__JPEG_ENCODER_HPP_
#define CAMERA_BASE__JPEG_ENCODER_HPP_

#include <NvJpegEncoder.h>

namespace cyberdog_camera
{

class JpegEncoder
{
public:
  JpegEncoder(int width, int height, const char *);
  ~JpegEncoder();

  void setCropRect(uint32_t left, uint32_t top, uint32_t width, uint32_t height);
  size_t encodeFromFd(int fd, unsigned char ** data, int quality = 75);

private:
  NvJPEGEncoder * m_jpegEncoder;
  uint32_t m_bufferSize;
  unsigned char * m_outputBuffer;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_BASE__JPEG_ENCODER_HPP_
