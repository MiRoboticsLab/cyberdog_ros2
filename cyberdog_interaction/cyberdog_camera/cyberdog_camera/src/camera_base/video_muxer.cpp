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

#include "camera_base/video_muxer.hpp"

namespace cyberdog_camera
{

uint8_t * extra_data;

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

VideoMuxer * VideoMuxer::create(int width, int height, const char * filename)
{
  VideoMuxer * muxer = new VideoMuxer();

  if (!muxer->initialize(width, height, filename)) {
    delete muxer;
    muxer = NULL;
  }

  return muxer;
}

VideoMuxer::VideoMuxer()
: m_frameCount(0),
  m_firstTs(0),
  m_lastTs(0)
{
  m_avContext = NULL;
  m_avStream = NULL;
}

VideoMuxer::~VideoMuxer()
{
}

bool VideoMuxer::initialize(int width, int height, const char * filename)
{
  AVOutputFormat * ofmt = NULL;
  AVCodec * out_encoder = NULL;

  extra_data = new uint8_t[32 * 1024];

  av_register_all();
  avformat_alloc_output_context2(&m_avContext, NULL, NULL, filename);
  if (!m_avContext) {
    printf("Failed to create output context!\n");
    return false;
  }

  ofmt = m_avContext->oformat;
  out_encoder = avcodec_find_encoder(AV_CODEC_ID_H264);
  m_avStream = avformat_new_stream(m_avContext, out_encoder);
  m_avStream->codecpar->codec_id = AV_CODEC_ID_H264;
  m_avStream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  m_avStream->codecpar->codec_tag = 0;
  m_avStream->codecpar->bit_rate = 10000000;
  m_avStream->codecpar->format = AV_PIX_FMT_YUV420P;
  m_avStream->codecpar->width = width;
  m_avStream->codecpar->height = height;

  av_dump_format(m_avContext, 0, filename, 1);

  /*open output file*/
  if (!(ofmt->flags & AVFMT_NOFILE)) {
    if (avio_open(&m_avContext->pb, filename, AVIO_FLAG_WRITE) < 0) {
      printf("Failed to open output file!%s\n", filename);
      return false;
    }
  }

  /*write file header*/
  if (avformat_write_header(m_avContext, NULL) < 0) {
    printf("Failed to write mp4 header!\n");
    return false;
  }

  return true;
}

void VideoMuxer::release()
{
  AVOutputFormat * ofmt = m_avContext->oformat;

  av_write_trailer(m_avContext);

  if (m_avContext && !(ofmt->flags & AVFMT_NOFILE)) {
    avio_close(m_avContext->pb);
  }

  avformat_free_context(m_avContext);
}

bool VideoMuxer::processData(uint8_t * data, size_t size, int64_t timestamp)
{
  int ret = 0;

  if (size == 0) {
    return true;
  }

  AVPacket pkt;
  uint8_t frame_type = data[4] & 0x1f;
  int64_t now_ms = 0;
  int stream_index = 0;

  if (m_frameCount == 0) {
    m_firstTs = timestamp;
    m_lastTs = timestamp;
  }

  /*sps & pps nal*/
  if (frame_type == 0x7) {
    m_avStream->codecpar->extradata_size = size;
    m_avStream->codecpar->extradata =
      reinterpret_cast<uint8_t *>(av_mallocz(size + AV_INPUT_BUFFER_PADDING_SIZE));
    memcpy(m_avStream->codecpar->extradata, data, size);
  }

  if (timestamp >= m_firstTs && timestamp >= m_lastTs) {
    now_ms = timestamp - m_firstTs;
    m_lastTs = timestamp;

    av_init_packet(&pkt);
    pkt.data = data;
    pkt.size = size;

    if (is_key_frame(frame_type)) {
      pkt.flags |= AV_PKT_FLAG_KEY;
    }

    pkt.pts = static_cast<double>(now_ms) /
      static_cast<double>(av_q2d(m_avStream->time_base) * 1000 * 1000);
    pkt.dts = pkt.pts;
    pkt.duration = 0;
    pkt.pos = -1;
    pkt.stream_index = stream_index;

    ret = av_interleaved_write_frame(m_avContext, &pkt);
    if (ret < 0) {
      printf("Failed to write frame, ret = %d\n", ret);
      return false;
    }
    av_packet_unref(&pkt);
  }

  m_frameCount++;
  return true;
}

}  // namespace cyberdog_camera
