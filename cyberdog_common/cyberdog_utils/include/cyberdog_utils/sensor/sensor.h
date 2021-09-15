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

#ifndef CYBERDOG_UTILS__SENSOR__SENSOR_H_
#define CYBERDOG_UTILS__SENSOR__SENSOR_H_

#include "cyberdog_utils/sensor/sensor_base.h"

/**
 * sensor event data
 */
typedef struct sensors_vec_t
{
  float v[3];
  unsigned int status;
} sensors_vec_t;

/**
 * uncalibrated accelerometer, gyroscope and magnetometer event data
 */
typedef struct uncalibrated_event_t
{
  float uncalib[3];
  float bias[3];
} uncalibrated_event_t;

/**
 * Union of the various types of sensor data
 * that can be returned.
 */
typedef struct sensors_event_t
{
  /* sensor type */
  unsigned int sensor_type;

  /* sensor data accuracy */
  unsigned int accuracy;

  /* time is in nanosecond */
  uint64_t timestamp;

  union {
    union {
      float data[16];

      /* acceleration values are in meter per second per second (m/s^2) */
      sensors_vec_t acceleration;

      /* magnetic vector values are in micro-Tesla (uT) */
      sensors_vec_t magnetic;

      /* orientation values are in degrees */
      sensors_vec_t orientation;

      /* gyroscope values are in rad/s */
      sensors_vec_t gyro;

      /* temperature is in degrees centigrade (Celsius) */
      float temperature;

      /* distance in centimeters */
      float distance;

      /* light in SI lux units */
      float light;

      /* pressure in hectopascal (hPa) */
      float pressure;

      /* relative humidity in percent */
      float relative_humidity;

      /* uncalibrated gyroscope values are in rad/s */
      uncalibrated_event_t uncalibrated_gyro;

      /* uncalibrated magnetometer values are in micro-Teslas */
      uncalibrated_event_t uncalibrated_magnetic;

      /* uncalibrated accelerometer values are in  meter per second per second (m/s^2) */
      uncalibrated_event_t uncalibrated_accelerometer;
    } vec;
  } sensor_data_t;
} sensors_event_t;
typedef struct config_event_t
{
  /* sensor type */
  uint8_t sensor_type;
  sensor_config_event_type config_type;
  union {
    uint32_t synchronized_timestamp;
    float config_data[6];
    float resp_data[6];
  } cfg_data;
} config_event_t;

typedef config_event_t resp_event_t;

typedef struct timer_event_t
{
  int timer_num;
  uint32_t synchronized_timestamp;
} timer_event_t;

typedef struct interrupt_event_t
{
  int interrupt_num;
  uint64_t synchronized_timestamp;
} interrupt_event_t;

typedef struct sensor_message_event_t
{
  sensor_message_event_type message_event_type;
  union {
    sensors_event_t sensor_data_event;
    config_event_t config_event;
    timer_event_t timer_event;
    interrupt_event_t interrupt_event;
    resp_event_t resp_event;
  } message_event_t;
} sensor_message_event_t;

#endif  // CYBERDOG_UTILS__SENSOR__SENSOR_H_
