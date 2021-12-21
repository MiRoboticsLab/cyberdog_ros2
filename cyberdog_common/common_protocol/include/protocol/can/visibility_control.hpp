// Copyright (c) 2021 Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
// Copyright 2021 the Autoware Foundation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Co-developed by Tier IV, Inc. and Apex.AI, Inc.

#ifndef PROTOCOL__CAN__VISIBILITY_CONTROL_HPP_
#define PROTOCOL__CAN__VISIBILITY_CONTROL_HPP_

// This logic was borrowed (then namespaced) from the examples on the gcc wiki:
//     https://gcc.gnu.org/wiki/Visibility

#if defined _WIN32 || defined __CYGWIN__
  #ifdef __GNUC__
    #define SOCKETCAN_EXPORT __attribute__ ((dllexport))
    #define SOCKETCAN_IMPORT __attribute__ ((dllimport))
  #else
    #define SOCKETCAN_EXPORT __declspec(dllexport)
    #define SOCKETCAN_IMPORT __declspec(dllimport)
  #endif
  #ifdef SOCKETCAN_BUILDING_LIBRARY
    #define SOCKETCAN_PUBLIC SOCKETCAN_EXPORT
  #else
    #define SOCKETCAN_PUBLIC SOCKETCAN_IMPORT
  #endif
  #define SOCKETCAN_PUBLIC_TYPE SOCKETCAN_PUBLIC
  #define SOCKETCAN_LOCAL
#else
  #define SOCKETCAN_EXPORT __attribute__ ((visibility("default")))
  #define SOCKETCAN_IMPORT
  #if __GNUC__ >= 4
    #define SOCKETCAN_PUBLIC __attribute__ ((visibility("default")))
    #define SOCKETCAN_LOCAL  __attribute__ ((visibility("hidden")))
  #else
    #define SOCKETCAN_PUBLIC
    #define SOCKETCAN_LOCAL
  #endif
  #define SOCKETCAN_PUBLIC_TYPE
#endif

#endif  // PROTOCOL__CAN__VISIBILITY_CONTROL_HPP_
