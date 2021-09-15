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

// This file stays the same with STM32 sensor code

#ifndef CYBERDOG_UTILS__SENSOR__SENSOR_BASE_H_
#define CYBERDOG_UTILS__SENSOR__SENSOR_BASE_H_

enum
{
  SENSOR_TYPE_ACCELEROMETER = 0,
  SENSOR_TYPE_MAGNETIC_FIELD = 1,
  SENSOR_TYPE_GYROSCOPE = 2,
  SENSOR_TYPE_LIGHT = 3,
  SENSOR_TYPE_LED_HEAD = 4,
  SENSOR_TYPE_PROXIMITY_HEAD = 5,
  SENSOR_TYPE_PROXIMITY_BOT = 6,
  SENSOR_TYPE_PROXIMITY_REAR = 7,
  SENSOR_TYPE_LIGHT_SPEED = 8,
  SENSOR_TYPE_LED_REAR = 9,
  // make sure virtual sensor index starts after all hardware sensors
  // we need to check if hw sensor present to decide if this virtual sensor should probe
  SENSOR_TYPE_ROTATION_VECTOR = 13,
  SENSOR_TYPE_SPEED_VECTOR = 14,
  SENSOR_TYPE_MAX = 15,
};

enum
{
  SENSOR_STATUS_NO_CONTACT = -1 /* -1 */,
  SENSOR_STATUS_UNRELIABLE = 0,
  SENSOR_STATUS_ACCURACY_LOW = 1,
  SENSOR_STATUS_ACCURACY_MEDIUM = 2,
  SENSOR_STATUS_ACCURACY_HIGH = 3,
};

typedef enum
{
  SENSOR_CONFIG_MESSAGE = 0,
  SENSOR_DATA_MESSAGE = 1,
  SENSOR_TIMESTAMP_MESSAGE = 2,
  SENSOR_CONFIG_RESP_MESSAGE = 3,
  SENSOR_OTA_MESSAGE = 4,
  SENSOR_VERSION_MSG = 5,
  SENSOR_TIME_SYNC_MSG = 6,
  SENSOR_DEBUG_CONFIG_MSG = 7,
  /* ----- can stdid supporting max to 7 here for external conmnication ----- */
  SENSOR_TIMER_EVENT = 8,
  SENSOR_INTERRUPT_EVENT = 9,
  SENSOR_INIT_COMPLETE_EVENT = 10,
  SENSOR_ACTIVATE_COMPLETE_EVENT = 11,
  SENSOR_DEACTIVATE_COMPLETE_EVENT = 12,
} sensor_message_event_type;

typedef enum
{
  SENSOR_DEACTIVATE = 0,
  SENSOR_ACTIVATE = 1,
  SENSOR_CONFIG_SELFTEST = 2,
  SENSOR_CONFIG_CALIBRATION = 3,
  SENSOR_CONFIG_BIAS = 4,
  SENSOR_CONFIG_META = 5,
  SENSOR_CONFIG_TIMEOUT = 6,
  SENSOR_CONFIG_DATA = 7,
  SENSOR_CALIBRATION_RESULT = 8,
} sensor_config_event_type;

/**
 * Values returned by the accelerometer in various locations in the universe.
 * all values are in SI units (m/s^2)
 */
#define GRAVITY_SUN                      (275.0f)
#define GRAVITY_EARTH              (9.80665f)

/** Maximum magnetic field on Earth's surface */
#define MAGNETIC_FIELD_EARTH_MAX        (60.0f)

/** Minimum magnetic field on Earth's surface */
#define MAGNETIC_FIELD_EARTH_MIN        (30.0f)

#define MAX_AUTO_DETECT_DEV_NUM 2

#endif  // CYBERDOG_UTILS__SENSOR__SENSOR_BASE_H_
