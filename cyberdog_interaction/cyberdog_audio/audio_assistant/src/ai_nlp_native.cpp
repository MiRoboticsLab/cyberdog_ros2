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

#define TRANS_TO_INSTRUCTION 1
#define AI_HOT_QUERY_CONF "/opt/ros2/cyberdog/ai_conf/hot_query_robotcontroller.txt"

#include "xiaoai_sdk/minerva/minerva_engine.h"
#ifdef TRANS_TO_INSTRUCTION
#include "xiaoai_sdk/aivs/Instruction.h"
#include "xiaoai_sdk/aivs/AllEnums.h"
#include "xiaoai_sdk/aivs/RobotController.h"
#endif
#include <string>
#include <cstdio>
#include <memory>

/*
enum class RobotAction {
    STANDING,
    DOWN,
    COME,
    BACK,
    GO_ROUND,
    HIGH_FIVE,
    BACK_SOMERSAULT,
    DANCE
};
*/
int ai_push_msg(int msg);

int ai_nlp_native_setup_check(const char * xcmd)
{
  int errornum;
  void * engine = minerva_engine_init(AI_HOT_QUERY_CONF, &errornum);
  printf("vc: init engine %d\n", errornum);
  if (engine) {
    // debug
    // nlp_result *result = minerva_engine_match(engine, "站起来");
    printf("vc: NLP ...  %s\n\n\n\n", xcmd);
    nlp_result * result = minerva_engine_match(engine, xcmd);

    if (result) {
      printf("vc: matched query:%s\n", result->raw_query);
      printf("vc: matched domain:%s %ld\n", result->domain, result->instructions_num);
#ifdef TRANS_TO_INSTRUCTION
      for (uint32_t i = 0; i < result->instructions_num; ++i) {
        /*    PROCESS INSCTRUCTIONS    */
        std::shared_ptr<aivs::Instruction> instruction;
        std::string json = result->instructions[i];
        if (aivs::Instruction::build(json, instruction)) {
          std::string inst_namespace = instruction->getHeader()->getNamespace();
          std::string inst_name = instruction->getHeader()->getName();
          if (inst_namespace == "RobotController" && inst_name == "Operate") {
            auto payload = std::static_pointer_cast<aivs::RobotController::Operate>(
              instruction->getPayload());
            aivs::RobotController::RobotAction action = payload->getAction();

            printf("vc: get Robot action:%d\n", static_cast<int>(action));
            int message = -1;
            switch (action) {
              case aivs::RobotController::RobotAction::STANDING:
                message = 1;
                break;
              case aivs::RobotController::RobotAction::DOWN:
                message = 2;
                break;
              case aivs::RobotController::RobotAction::COME:
                message = 3;
                break;
              case aivs::RobotController::RobotAction::BACK:
                message = 4;
                break;
              case aivs::RobotController::RobotAction::GO_ROUND:
                message = 5;
                break;
              case aivs::RobotController::RobotAction::HIGH_FIVE:
                message = 6;
                break;
              case aivs::RobotController::RobotAction::BACK_SOMERSAULT:
                message = -1;  // not support
                break;
              case aivs::RobotController::RobotAction::DANCE:
                message = 7;
                break;
              default:
                break;
            }
            ai_push_msg(message);
          }
        }
      }
#endif
    } else {
      printf("vc: no matched\n");
    }
    // void minerva_engine_free_result(nlp_result *result);
    minerva_engine_free_result(result);
    // void minerva_engine_release(void *engine);
    minerva_engine_release(engine);
  }
  return 0;
}
