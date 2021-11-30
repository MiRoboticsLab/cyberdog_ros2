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

#ifndef CAMERA_BASE__CAMERA_DISPATCHER_HPP_
#define CAMERA_BASE__CAMERA_DISPATCHER_HPP_

#include <Argus/Argus.h>
#include <vector>

namespace cyberdog_camera
{
using Argus::CaptureSession;
using Argus::CameraDevice;
using Argus::CameraProvider;
using Argus::Request;
using Argus::OutputStream;
using Argus::UniqueObj;
using Argus::Size2D;

class CameraDispatcher
{
public:
  static CameraDispatcher & getInstance();
  bool createSession(UniqueObj<CaptureSession> & session, uint32_t deviceIndex);
  bool createOutputStream(
    CaptureSession * session,
    UniqueObj<OutputStream> & stream, Size2D<uint32_t> size,
    Argus::StreamType type = Argus::STREAM_TYPE_BUFFER);
  bool createRequest(
    CaptureSession * session,
    UniqueObj<Request> & request, Argus::CaptureIntent captureIntent);
  bool enableOutputStream(Request * request, OutputStream * stream);
  bool disableOutputStream(Request * request, OutputStream * stream);
  bool clearOutputStreams(Request * request);
  bool getOutputStreams(Request * request, std::vector<OutputStream *> * streams);
  bool capture(CaptureSession * session, Request * request);
  bool startRepeat(CaptureSession * session, Request * request);
  bool stopRepeat(CaptureSession * session);
  bool waitForIdle(CaptureSession * session);
  bool isRepeating(CaptureSession * session);
  bool cancelRequests(CaptureSession * session);
  Size2D<uint32_t> getSensorSize(uint32_t deviceIndex);

private:
  CameraDispatcher();
  ~CameraDispatcher();
  bool initialize();
  bool shutdown();

  bool m_initialized;
  UniqueObj<CameraProvider> m_cameraProvider;
  Argus::ICameraProvider * m_iCameraProvider;
  std::vector<CameraDevice *> m_cameraDevices;
};

}  // namespace cyberdog_camera

#endif  // CAMERA_BASE__CAMERA_DISPATCHER_HPP_
