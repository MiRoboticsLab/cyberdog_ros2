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

#ifndef CAMERA_BASE__STREAM_CONSUMER_HPP_
#define CAMERA_BASE__STREAM_CONSUMER_HPP_

#include <Argus/Argus.h>
#include <EGLStream/EGLStream.h>
#include <EGLStream/NV/ImageNativeBuffer.h>
#include <Thread.h>
#include <map>
#include "./camera_buffer.hpp"

#define MAX_BUFFERS_NUM 8

namespace cyberdog_camera
{
using Argus::Size2D;
using Argus::OutputStream;

class StreamConsumer : public ArgusSamples::Thread
{
public:
  StreamConsumer(
    Size2D<uint32_t> size,
    NvBufferLayout layout = NvBufferLayout_Pitch,
    int num_buf = MAX_BUFFERS_NUM);
  virtual ~StreamConsumer();

  void setOutputStream(OutputStream * stream)
  {
    m_outputStream.reset(stream);
  }

  OutputStream * getOutputStream()
  {
    return m_outputStream.get();
  }

  Size2D<uint32_t> getSize()
  {
    return m_size;
  }

  bool endOfStream();

protected:
  Argus::Buffer * getBufferFromFd(int fd)
  {
    return m_bufferMap[fd];
  }
  bool getFrameRate(float & frameRate);

  Argus::UniqueObj<OutputStream> m_outputStream;
  Size2D<uint32_t> m_size;
  NvBufferLayout m_layout;
  int m_maxBuffersNum;
  uint64_t m_frameCount;
  uint64_t m_firstFrameTime;
  uint32_t m_fps;

  Argus::UniqueObj<Argus::Buffer> m_buffers[MAX_BUFFERS_NUM];
  DmaBuffer * m_nativeBuffers[MAX_BUFFERS_NUM];
  EGLDisplay m_eglDisplay;
  EGLImageKHR m_eglImages[MAX_BUFFERS_NUM];
  std::map<int, Argus::Buffer *> m_bufferMap;

  //  Thread methods
  virtual bool threadInitialize();
  virtual bool threadExecute();
  virtual bool threadShutdown();

  virtual bool processBuffer(Buffer * buffer) = 0;
  bool bufferDone(Buffer * buffer);
};

}  // namespace cyberdog_camera
#endif  // CAMERA_BASE__STREAM_CONSUMER_HPP_
