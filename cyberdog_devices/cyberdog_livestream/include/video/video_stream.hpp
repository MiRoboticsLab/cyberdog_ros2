// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
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

#ifndef VIDEO__VIDEO_STREAM_HPP_
#define VIDEO__VIDEO_STREAM_HPP_

#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavformat/avio.h>
#include <libswscale/swscale.h>
}

#include <string>
#include <queue>
#include <mutex>
#include <vector>
#include <thread>
#include <condition_variable>
#include <iostream>
#include <memory>

#include "./log.hpp"

namespace live_stream
{

static void getCurrentTime()
{
  uint64_t timestamp(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().
      time_since_epoch()).count());
  uint64_t milli = timestamp + 8 * 60 * 60 * 1000;
  auto mTime = std::chrono::milliseconds(milli);
  auto tp = std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds>(mTime);
  const auto tt = std::chrono::system_clock::to_time_t(tp);
  tm now;
  gmtime_r(&tt, &now);
  std::cout << getpid() << ":" << std::this_thread::get_id() << "    " << now.tm_hour << ":" << \
    now.tm_min << ":" << now.tm_sec << "." << int(timestamp % 1000) << " ";
}

class VideoStream
{
public:
  VideoStream() {}
  explicit VideoStream(const std::string & dest);
  ~VideoStream()
  {
    reset();
  }
  void reset()
  {
    std::lock_guard<std::mutex> lk(mActiveMutex);
    if (mWork != nullptr) {
      mWork.reset();
      mWork = nullptr;
    }

    mInitCodec = false;
    mInitWH = false;
    mIsActive = false;

    if (mFrame != NULL) {
      av_frame_free(&mFrame);
      mFrame = NULL;
    }
    if (mPacket != NULL) {
      av_packet_unref(mPacket);
      mPacket = NULL;
    }
    if (mCodecContext != NULL) {
      avcodec_free_context(&mCodecContext);
      mCodecContext = NULL;
    }
    if (mSwsContext != NULL) {
      sws_freeContext(mSwsContext);
      mSwsContext = NULL;
    }
    if (mOutputFormat != NULL) {
      avio_close(mOutputFormat->pb);
      avformat_free_context(mOutputFormat);
      mOutputFormat = NULL;
    }
  }

private:
  AVFormatContext * mOutputFormat = NULL;
  AVCodecContext * mCodecContext = NULL;
  SwsContext * mSwsContext = NULL;
  AVStream * mVideoStream;

  bool mInitCodec = false;
  bool mInitWH = false;
  int mSrcWidth;
  int mSrcHeight;
  int mDesWidth;
  int mDesHeight;
  AVPixelFormat mSrcPixFmt;

  AVFrame * mFrame = NULL;
  int64_t mPts = 0;
  AVPacket * mPacket = NULL;

  std::string mDestUrl;
  bool mIsActive;
  std::mutex mActiveMutex;

  class Work;
  std::shared_ptr<Work> mWork = nullptr;
  uint64_t mStartTs;
  uint64_t mFrameNum;

public:
  std::thread mStreamThread;
  std::mutex mThreadMutex;
  std::condition_variable mCondition;
  std::queue<std::shared_ptr<std::vector<uint8_t>>> mBuffers;
  std::mutex mBufferMutex;

  void setInputSize(int width, int height)
  {
    mSrcWidth = width;
    mSrcHeight = height;
  }

  void setOutSize(int width, int height)
  {
    mDesWidth = width;
    mDesHeight = height;
  }

  bool isActive()
  {
    return mIsActive;
  }

  int emptyThisBuffer(const uint8_t * data, uint32_t size);
  int emptyThisBuffer(uint8_t * data, uint32_t size, uint64_t ts);
  void startWork();
  int initDecoder();
  void configure(int w = 1280, int h = 960, AVPixelFormat fmt = AV_PIX_FMT_NV12);
  void sendImage(const std::vector<uint8_t> & data, const uint64_t time);
};
}  // namespace live_stream
#endif  // VIDEO__VIDEO_STREAM_HPP_
