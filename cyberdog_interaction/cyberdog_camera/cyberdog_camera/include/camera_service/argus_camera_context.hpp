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

#ifndef CAMERA_SERVICE__ARGUS_CAMERA_CONTEXT_HPP_
#define CAMERA_SERVICE__ARGUS_CAMERA_CONTEXT_HPP_

#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include "camera_base/stream_consumer.hpp"

namespace cyberdog_camera
{

enum Streamtype
{
  STREAM_TYPE_ALGO = 0,
  STREAM_TYPE_RGB,
  STREAM_TYPE_VIDEO,
  STREAM_TYPE_H264,
  STREAM_TYPE_MAX,
};

static const unsigned MAX_STREAM = 5;

class ArgusCameraContext
{
public:
  explicit ArgusCameraContext(int camId);
  ~ArgusCameraContext();

  bool createSession();
  bool closeSession();
  bool initCameraContext();
  bool deinitCameraContext();
  int takePicture(const char * path, int width, int height);
  int startPreview(std::string & usage);
  int stopPreview();
  int startRecording(std::string & filename, int width, int height);
  int stopRecording(std::string & filename);
  int startRgbStream();
  int stopRgbStream();
  bool isRecording();
  uint64_t getRecordingTime();

  Size2D<uint32_t> getSensorSize();

private:
  bool initialize();
  bool shutdown();
  int startCameraStream(Streamtype type, Size2D<uint32_t> size, void * args);
  int stopCameraStream(Streamtype type);

  int m_cameraId;

  Argus::UniqueObj<Argus::CaptureSession> m_captureSession;
  Argus::UniqueObj<Argus::Request> m_previewRequest;

  std::string m_videoFilename;
  std::vector<std::shared_ptr<StreamConsumer>> m_activeStreams;
  std::mutex m_streamLock;
  bool m_isStreaming;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_SERVICE__ARGUS_CAMERA_CONTEXT_HPP_
