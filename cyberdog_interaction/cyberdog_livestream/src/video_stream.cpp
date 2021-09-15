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

#include <time.h>

#include <algorithm>
#include <chrono>
#include <utility>
#include <functional>
#include <memory>
#include <vector>
#include <string>
#include <queue>

#include "video/video_stream.hpp"

namespace live_stream
{

#define FPS 30
#define VIDEO_BUFFER_SIZE 12000000  // 3000x2000x2
class VideoStream::Work
{
private:
  VideoStream * mOwner;
  std::thread mThread;
  bool mQuit;

public:
  explicit Work(VideoStream * parent)
  : mOwner(parent), mQuit(false) {}
  ~Work()
  {
    mQuit = true;
    if (mThread.joinable()) {
      mThread.join();
    }
  }
  void start()
  {
    std::cout << "==>will create thread" << std::endl;
    mThread = std::thread(std::bind(&Work::onWork, this));
  }
  void setQuit()
  {
    mQuit = true;
  }
  void onWork() const
  {
    auto ptr = &(mOwner->mBuffers);
    std::cout << "enter onWork" << std::endl;
    while (!mQuit) {
      getCurrentTime();
      std::cout << "==>enter onWork :" << ptr->size() << std::endl;
      std::unique_lock<std::mutex> lock(mOwner->mBufferMutex);
      /*
      while (ptr->empty())
      {   std::cout<<"to sleep"<<std::endl;
          mOwner->mCondition.wait_for(lock,std::chrono::milliseconds(33));
          std::cout<<"sleep after 30ms size:"<<ptr->size()<<std::endl;
      }
      */
      mOwner->mCondition.wait(lock, [&] {return !ptr->empty();});

      std::shared_ptr<std::vector<uint8_t>> frame;
      frame = std::move(ptr->front());
      ptr->pop();
      lock.unlock();
      mOwner->sendImage(*frame, 0);
    }
    std::cout << "thread id:" << mThread.get_id() << " quit" << std::endl;
  }
};

VideoStream::VideoStream(const std::string & dest)
{
  mDestUrl = dest;
  mWork = nullptr;
  av_register_all();
  avformat_network_init();
  mStartTs = 0;
  mFrameNum = 0;
}

void VideoStream::startWork()
{
  if (mWork != nullptr) {
    std::cout << "have launched works" << std::endl;
  }
  mWork = std::make_shared<Work>(this);
  mWork->start();
}
void VideoStream::configure(int w, int h, AVPixelFormat pix)
{
  if (mInitWH) {
    return;
  }
  mSrcWidth = w;
  mSrcHeight = h;
  mSrcPixFmt = pix;

  mDesWidth = 640;
  mDesHeight = 480;
  mInitWH = true;
}

int VideoStream::initDecoder()
{
  std::lock_guard<std::mutex> lk(mActiveMutex);
  if (mInitCodec) {
    std::cout << " decoder Inited " << std::endl;
    return -MEDIA_ERROR_CONFIG;
  }
  //   mSrcPixFmt = AV_PIX_FMT_NV12;
  int ret;
  AVDictionary * param = 0;

  av_dict_set(&param, "buffer_size", "0", 0);
  av_dict_set(&param, "rtbufsize", "0", 0);

  av_dict_set(&param, "preset", "superfast", 0);
  av_dict_set(&param, "tune", "zerolatency", 0);
  av_dict_set(&param, "crf", "25", 0);
  av_dict_set(&param, "threads", "2", 0);
  av_dict_set(&param, "fflags", "nobuffer", 0);

  std::cout << "dest W H" << mDesWidth << " * " << mDesHeight << "msrcWidth: " << \
    mSrcWidth << " mSrcHeight: " << mSrcHeight << std::endl;
  mSwsContext = sws_getCachedContext(
    mSwsContext, mSrcWidth, mSrcHeight, AV_PIX_FMT_NV12,
    mDesWidth, mDesHeight, AV_PIX_FMT_NV12,
    SWS_BICUBIC,
    0, 0, 0);

  if (mSwsContext == NULL) {
    std::cout << "fatal error can't alloc mSwsContext" << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }
  mFrame = av_frame_alloc();
  mFrame->format = AV_PIX_FMT_NV12;
  mFrame->width = mDesWidth;
  mFrame->height = mDesHeight;
  mFrame->pts = 0;

  ret = av_frame_get_buffer(mFrame, 32);
  if (ret != 0) {
    std::cout << "fatal error can't alloc yuv buffer" << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }

  AVCodec * codec = avcodec_find_encoder(AV_CODEC_ID_H264);
  if (codec == NULL) {
    std::cout << "fatal error can't find H264 encoder" << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }
  mCodecContext = avcodec_alloc_context3(codec);
  if (mCodecContext == NULL) {
    std::cout << "fatal error can't allocate H264 encoder" << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }

  mCodecContext->flags |= AV_CODEC_FLAG_GLOBAL_HEADER | AV_CODEC_FLAG_LOW_DELAY;
  mCodecContext->flags2 |= AV_CODEC_FLAG2_FAST;
  mCodecContext->profile = FF_PROFILE_H264_BASELINE;
  mCodecContext->codec_id = codec->id;
  mCodecContext->thread_count = 6;
  mCodecContext->bit_rate = 2.4 * 1024 * 1024;        // 2Mbits
  mCodecContext->width = mDesWidth;
  mCodecContext->height = mDesHeight;
  mCodecContext->framerate = {FPS, 1};
  mCodecContext->time_base = {1, FPS};
  mCodecContext->gop_size = 1;
  mCodecContext->max_b_frames = 0;
  mCodecContext->pix_fmt = AV_PIX_FMT_NV12;
  ret = avcodec_open2(mCodecContext, 0, 0);
  if (ret != 0) {
    std::cout << "codec can't open" << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }

  ret = avformat_alloc_output_context2(&mOutputFormat, NULL, "flv", mDestUrl.c_str());
  if (ret < 0) {
    std::cout << "avformat_alloc_output_context2 failed" << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }
  av_dict_copy(&mOutputFormat->metadata, param, AV_DICT_APPEND);

  mVideoStream = avformat_new_stream(mOutputFormat, NULL);
  if (mVideoStream == NULL) {
    std::cout << "fatal error,can't create avstream" << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }
  mVideoStream->codecpar->codec_tag = 0;
  avcodec_parameters_from_context(mVideoStream->codecpar, mCodecContext);
  mVideoStream->codecpar->bit_rate = 1 * 1024 * 1024;

  av_dump_format(mOutputFormat, 0, mDestUrl.c_str(), 1);

  ret = avio_open(&mOutputFormat->pb, mDestUrl.c_str(), AVIO_FLAG_WRITE);
  if (ret < 0) {
    std::cout << "fatal error,can't open: " << mDestUrl.c_str() << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }
  ret = avformat_write_header(mOutputFormat, NULL);
  if (ret < 0) {
    std::cout << "fatal error,can't init header about: " << mDestUrl.c_str() << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }

  mPacket = av_packet_alloc();
  if (mPacket == NULL) {
    std::cout << "fatal error,can't allocate outbuf " << std::endl;
    reset();
    return -MEDIA_ERROR_CONFIG;
  }

  mInitCodec = true;
  return MEDIA_OK;
}

int VideoStream::emptyThisBuffer(const uint8_t * data, uint32_t size)
{
  getCurrentTime();
  std::cout << ": =======>" << __func__ << "size:" << size << std::endl;
  if (mWork == nullptr /* ||!mIsActive*/) {
    std::cout << "fatal error:pushing stream working don't start" << std::endl;
    return -MEDIA_ERROR_UNKNOWN;
  }
  std::shared_ptr<std::vector<uint8_t>> buff = std::make_shared<std::vector<uint8_t>>(size);
  std::copy(data, data + size, buff->begin());
  {
    std::lock_guard<std::mutex> lk(mBufferMutex);
    if (!mBuffers.empty()) {
      std::cout << "warning: cache:" << mBuffers.size() << std::endl;
      std::queue<std::shared_ptr<std::vector<uint8_t>>> empty;
      mBuffers.swap(empty);
      // std::swap(empty,mBuffers);
    }
    mBuffers.push(buff);
  }
  mCondition.notify_all();
  return MEDIA_OK;
}

void VideoStream::sendImage(const std::vector<uint8_t> & data, const uint64_t time)
{
  getCurrentTime();
  std::cout << "==>enter VideoStream vector::" << __func__;
  std::cout << "\n";
  std::cout << "timestamp" << std::endl;
  if (data.empty()) {
    std::cout << "empty data" << time << std::endl;
    return;
  }
  int ret;
  uint8_t indata[VIDEO_BUFFER_SIZE];
  std::copy(data.begin(), data.end(), indata);
  uint8_t * yuvdata[2] = {indata, indata + mSrcWidth * mSrcHeight};
  int linesize[2] = {mSrcWidth, mSrcWidth};
  // uint8_t* linesize[2] = {mDesWidth,mDesWidth};
  ret = sws_scale(mSwsContext, yuvdata, linesize, 0, mSrcHeight, mFrame->data, mFrame->linesize);
  if (ret <= 0) {
    std::cout << "scale error" << std::endl;
  }
  mFrame->pts = mPts;
  mPts++;

  if (mInitCodec) {
    std::cout << "inited codec" << std::endl;
  } else {
    std::cout << "codec don't inited" << std::endl;
    return;
  }
  ret = avcodec_send_frame(mCodecContext, mFrame);
  if (ret != 0) {
    std::cout << "failed sending frame:" << mPacket->size << std::endl;
    return;
  }

  ret = avcodec_receive_packet(mCodecContext, mPacket);
  if (ret != 0) {
    std::cout << "failed encoding frame:" << mPacket->size << std::endl;
    return;
  }

  mPacket->pts = av_rescale_q(mPacket->pts, mCodecContext->time_base, mVideoStream->time_base);
  mPacket->dts = av_rescale_q(mPacket->dts, mCodecContext->time_base, mVideoStream->time_base);
  mPacket->duration = av_rescale_q(
    mPacket->duration, mCodecContext->time_base,
    mVideoStream->time_base);

  getCurrentTime();
  std::cout << "is pushing stream...." << std::endl;

  ret = av_interleaved_write_frame(mOutputFormat, mPacket);
  if (ret == 0) {
    getCurrentTime();
    std::cout << "push stream success" << std::endl;
  } else {
    getCurrentTime();
    std::cout << "push stream error" << std::endl;
  }
}

static inline bool is_key_frame(uint8_t type)
{
  switch (type) {
    case 7:
    case 8:
    case 5:
      return true;
      break;
    case 1:
      return false;
      break;
    default:
      return false;
  }

  return false;
}

int VideoStream::emptyThisBuffer(uint8_t * data, uint32_t size, uint64_t ts)
{
  int ret = 0;

  if (size == 0 || !mOutputFormat || !mPacket) {
    return -MEDIA_ERROR_UNKNOWN;
  }

  uint8_t frame_type = data[4] & 0x1f;
  if (mFrameNum == 0) {
    mStartTs = ts;
  }

  if (is_key_frame(frame_type)) {
    mPacket->flags |= AV_PKT_FLAG_KEY;
  }

  uint64_t now_ms = ts - mStartTs;
  mPacket->data = data;
  mPacket->size = size;
  mPacket->pts = static_cast<double>(now_ms) /
    static_cast<double>(av_q2d(mVideoStream->time_base) * 1000 * 1000);
  mPacket->dts = mPacket->pts;
  mPacket->pos = -1;
  mPacket->duration = av_rescale_q(
    mPacket->duration, mVideoStream->time_base,
    mVideoStream->time_base);

  ret = av_interleaved_write_frame(mOutputFormat, mPacket);
  if (ret < 0) {
    getCurrentTime();
    std::cout << "push stream error" << std::endl;
    return -MEDIA_ERROR_SERVER;
  }
  mFrameNum++;

  return MEDIA_OK;
}

}  // namespace live_stream
