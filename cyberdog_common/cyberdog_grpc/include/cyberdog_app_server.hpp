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


#ifndef CYBERDOG_APP_SERVER_HPP_
#define CYBERDOG_APP_SERVER_HPP_

#include <grpc/grpc.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>

#include "./cyberdog_app.grpc.pb.h"
#include "cyberdog_app.hpp"

using grpc::Server;
using grpc::ServerBuilder;
using grpc::ServerContext;
using grpc::ServerReader;
using grpc::ServerReaderWriter;
using grpc::ServerWriter;
using grpc::Status;
using std::chrono::system_clock;

class CyberdogAppImpl final : public cyberdogapp::CyberdogApp::Service
{
public:
  explicit CyberdogAppImpl(const std::string & db);
  void SetRequesProcess(cyberdog_cyberdog_app::Cyberdog_app * decision)
  {
    decision_ = decision;
  }
  ::grpc::Status setMode(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::CheckoutMode_request * request,
    ::grpc::ServerWriter<::cyberdogapp::CheckoutMode_respond> * writer)
  override;
  ::grpc::Status setPattern(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::CheckoutPattern_request * request,
    ::grpc::ServerWriter<::cyberdogapp::CheckoutPattern_respond> * writer)
  override;
  ::grpc::Status requestCamera(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::CameraService_request * request,
    ::grpc::ServerWriter<::cyberdogapp::CameraService_respond> * writer)
  override;
  ::grpc::Status sendAppDecision(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::Decissage * request,
    ::cyberdogapp::Result * response)
  override;

  ::grpc::Status setFollowRegion(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::BodyRegion_Request * request,
    ::grpc::ServerWriter<::cyberdogapp::BodyRegion_Respond> * writer)
  override;

  ::grpc::Status requestVoice(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::Voiceprint_Request * request,
    ::grpc::ServerWriter<::cyberdogapp::Voiceprint_Response> * writer)
  override;
  ::grpc::Status requestFaceManager(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::FaceManager_Request * request,
    ::grpc::ServerWriter<::cyberdogapp::FaceManager_Response> * writer)
  override;
  ::grpc::Status sendAiToken(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::TokenPass_Request * request,
    ::grpc::ServerWriter<::cyberdogapp::TokenPass_Response> * writer)
  override;
  ::grpc::Status setNavPosition(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::Target_Request * request,
    ::grpc::ServerWriter<::cyberdogapp::Target_Response> * writer)
  override;
  ::grpc::Status sendMotionTestRequest(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::MotionCommand * request,
    ::cyberdogapp::Result * response)
  override;

  ::grpc::Status getOffsetData(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::OffsetRequest * request,
    ::grpc::ServerWriter< ::cyberdogapp::OffsetCalibationData> * writer)
  override;

  ::grpc::Status setOffsetData(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::OffsetCalibationData * request,
    ::grpc::ServerWriter< ::cyberdogapp::OffsetRequest_result> * writer)
  override;

  ::grpc::Status setExtmonOrder(
    ::grpc::ServerContext * context,
    const ::cyberdogapp::ExtMonOrder_Request * request,
    ::grpc::ServerWriter< ::cyberdogapp::ExtMonOrder_Respond> * writer)
  override;

  ::grpc::Status disconnect(
    ::grpc::ServerContext * context, const ::cyberdogapp::Disconnect * request,
    ::grpc::ServerWriter<::cyberdogapp::Result> * writer)
  override;

  ::grpc::Status setBtRemoteCmd(
    ::grpc::ServerContext * context, const ::cyberdogapp::BtRemoteCommand_Request * request,
    ::grpc::ServerWriter<::cyberdogapp::BtRemoteCommand_Respond> * writer)
  override;

  ::grpc::Status setBodyPara(
    ::grpc::ServerContext * context, const ::cyberdogapp::Parameters * request,
    ::cyberdogapp::Result * response)
  override;

private:
  void process_motion(const ::cyberdogapp::MotionCommand * request);
  cyberdog_cyberdog_app::Cyberdog_app * decision_;
};

#endif  // CYBERDOG_APP_SERVER_HPP_
