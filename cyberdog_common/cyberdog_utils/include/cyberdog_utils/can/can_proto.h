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

// This file is used for can communication on STM32 can bus

#ifndef CYBERDOG_UTILS__CAN__CAN_PROTO_H_
#define CYBERDOG_UTILS__CAN__CAN_PROTO_H_

// TxHeader.StdId: (EVENT_TYPE 1byte) << 8 | (SENSOR_TYPE 1byte) << 4 | (COMMAND_CODE 1byte)
// TxHeader.ExtId = 0x01;

#define SENSOR_TYPE_BIT_SHIFT                 4
#define SENSOR_TYPE_BIT_MASK                  0x0F << SENSOR_TYPE_BIT_SHIFT
#define SENSOR_EVENT_MESSAGE_BIT_SHIFT        8
#define SENSOR_EVENT_MESSAGE_BIT_MASK         0x0F << SENSOR_EVENT_MESSAGE_BIT_SHIFT
#define SENSOR_COMMAND_BIT_MASK               0x0F

// SENSOR_CONFIG_MESSAGE      | SENSOR_TYPE_MAX            | SENSOR_ACTIVATE
#define CAN_ID_SENSOR_ENABLE_ALL          0x0F1
// SENSOR_CONFIG_MESSAGE      | SENSOR_TYPE_MAX            | SENSOR_DEACTIVATE
#define CAN_ID_SENSOR_DISABLE_ALL         0x0F0
// SENSOR_CONFIG_MESSAGE      | SENSOR_TYPE_ACCELEROMETER  | SENSOR_CONFIG_SELFTEST
#define CAN_ID_SENSOR_SELFTEST_ACC        0x002
// SENSOR_DATA_MESSAGE        | SENSOR_TYPE_ACCELEROMETER  | DATA_AXIS_NUM
#define CAN_ID_SENSOR_ACC_DATA            0x103
// SENSOR_DATA_MESSAGE        | SENSOR_TYPE_MAGNETIC_FIELD | DATA_AXIS_NUM
#define CAN_ID_SENSOR_MAG_DATA            0x113
// SENSOR_DATA_MESSAGE        | SENSOR_TYPE_GYROSCOPE      | DATA_AXIS_NUM
#define CAN_ID_SENSOR_GYRO_DATA           0x123
// SENSOR_TIMESTAMP_MESSAGE   | SENSOR_TYPE_GYROSCOPE      | DATA_AXIS_NUM
#define CAN_ID_SENSOR_GYRO_TINESTAMP      0x823


#endif  // CYBERDOG_UTILS__CAN__CAN_PROTO_H_
