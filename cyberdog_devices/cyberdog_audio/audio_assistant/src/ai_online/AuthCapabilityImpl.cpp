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

#include <iostream>
#include <fstream>
#include <string>
#include <memory>

#include "xiaoai_sdk/aivs/AuthCapabilityImpl.h"

static bool tokenRequest = false;
int ai_push_msg(int msg);

/**
 * 使用APP OAuth或device OAuth时，SDK需要从客户端获取auth code
 */
std::string AuthCapabilityImpl::onGetAuthCode()
{
  // auth code 有过期时间，且只能用一次; 每次回调请使用账号SDK获取新的auth code再返回
  std::string authCode = "C3_9758F778B806EBB494B567C66006FE87";
  std::cout << "vc: authCode:" << authCode << std::endl;
  return authCode;
}

void AuthCapabilityImpl::onAuthStateChanged(AuthState authState)
{
  std::cout << "vc: authState:" << authState << std::endl;
}

/**
 * sdk通过onGetToken从客户端拿token，token的获取和刷新由客户端自己负责；
 * 鉴权方式为ENGINE_AUTH_MIOT时必须实现，返回token组成为：session_id:xxx,token:xxx;
 * 其余鉴权方式当设置KEY_OBTAIN_TOKEN_IN_SDK为false时必须实现，直接返回access_token原文；
 */
bool AuthCapabilityImpl::onGetToken(int, bool, std::string & accessToken)
{
  accessToken = "session_id:65880_123244346_1547520112960346552,token:d8lbq9out0";
  return true;
}

static std::shared_ptr<std::string> token_access_aivs;
static std::shared_ptr<std::string> token_refresh_aivs;
void updateToken2Aivs(std::string token_access, std::string token_refresh)
{
  token_access_aivs = std::make_shared<std::string>(token_access.c_str());
  token_refresh_aivs = std::make_shared<std::string>(token_refresh.c_str());
  std::cout << "vc: token_access_aivs: " << token_access_aivs->c_str() << std::endl;
  std::cout << "vc: token_refresh_aivs: " << token_refresh_aivs->c_str() << std::endl;
}

/**
 * sdk通过onGetAuthorizationTokens从客户端拿access_token、refresh_token、expireIn；
 * 混合鉴权客户端需要实现的，客户端负责获取token，SDK负责刷新token，需要在回调的时候，客户端拿到token，封装成AuthorizationTokens返回给SDK
 * 目前只有Oauth鉴权支持混合模式，且需要客户端设置混合模式配置项aivs::AivsConfig::Auth::REQ_TOKEN_HYBRID为true
 * 示例代码中的accessToken、refreshToken、expireIn仅提供参考，客户端需要实际情况返回真实有效值
 */

std::shared_ptr<AuthCapability::AuthorizationTokens> AuthCapabilityImpl::onGetAuthorizationTokens()
{
  std::shared_ptr<AuthCapability::AuthorizationTokens> authorizationTokens =
    std::make_shared<AuthCapability::AuthorizationTokens>();
  authorizationTokens->accessToken = token_access_aivs->c_str();
  authorizationTokens->refreshToken = token_refresh_aivs->c_str();
  authorizationTokens->expireIn = 2592000;

  std::cout << "vc: online sdk call onGetAuthorizationTokens [AAAAA]" << std::endl;
  if (tokenRequest == false) {
    ai_push_msg(110);
    tokenRequest = true;
  }

  return authorizationTokens;
}
