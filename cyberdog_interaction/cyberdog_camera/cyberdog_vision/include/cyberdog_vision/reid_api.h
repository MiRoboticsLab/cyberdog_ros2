// Copyright (c) 2021  Beijing Xiaomi Mobile Software Co., Ltd. All rights reserved.
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

#ifndef CYBERDOG_VISION__REID_API_H_
#define CYBERDOG_VISION__REID_API_H_

#include <vector>
#include "./algorithm.h"

/**
 * @brief  Features match method.
 **/
typedef enum
{
  MIN_DIST_MAX_SIM = 1,  // max similarity
  MEAN_DIST_SIM,         // mean similarity
  MEAN_FEAT,             // mean features
} GroupMatchMethod;

/**
 * @brief  Features type.
 **/
typedef std::vector < float > FeaturesType;

/**
 * @brief  Reid algorithm initialize.
 * @return Handle, NULL if init failed.
 **/
AlgoHandle reid_init();

/**
 * @brief  Reid algorithm destroy.
 * @param  handle  Algo handle from reid_init().
 **/
void reid_destroy(AlgoHandle handle);

/**
 * @brief  Extract features from a image.
 * @param  handle  Algorithm handle.
 * @param  buffer  Image buffer.
 * @param  features  Output features.
 * @return false if extract failed.
 **/
bool reid_feature_extract(
  AlgoHandle handle,
  BufferInfo * buffer,
  FeaturesType & features);

/**
 * @brief  Get size of features.
 * @return Size of features.
 **/
int reid_get_feature_size();

/**
 * @brief  Get similarity of features one to one.
 * @param  handle  Algorithm handle.
 * @param  feats  Features to be compared.
 * @param  probe  Another features to be compared.
 * @return  similarity.
 **/
double reid_similarity_one2one(
  AlgoHandle handle,
  FeaturesType & feats,
  FeaturesType & probe);

/**
 * @brief  Get similarity of features one to group.
 * @param  handle  Algorithm handle.
 * @param  feats  Features to be compared.
 * @param  probe  Features group to be compared.
 * @param  num_probe  Features size in group.
 * @param  method  Match method.
 * @return  similarity.
 **/
double reid_similarity_one2group(
  AlgoHandle handle,
  FeaturesType & feats,
  FeaturesType & probe,
  int num_probe, GroupMatchMethod method = MEAN_FEAT);

/**
 * @brief  Get similarity of features one to group.
 * @param  handle  Algorithm handle.
 * @param  feats  Features group to be compared.
 * @param  num_feats  Features size in group.
 * @param  probe  Features group to be compared.
 * @param  num_probe  Features size in group.
 * @param  method  Match method.
 * @return  similarity.
 **/
double reid_similarity_group2group(
  AlgoHandle handle,
  FeaturesType & feats, int num_feats,
  FeaturesType & probe, int num_probe,
  GroupMatchMethod method = MEAN_FEAT);

/**
 * @brief  Judge same or not by similarity.
 * @return True if same.
 **/
bool reid_match_by_similarity(double similarity);

#endif  // CYBERDOG_VISION__REID_API_H_
