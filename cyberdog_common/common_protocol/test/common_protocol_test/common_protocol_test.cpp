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

#include <string>
#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "common_protocol/common_protocol.hpp"

#define EVM cyberdog::common
#define CLCT clct.GetAllStateTimesNum

class testing_full_var
{
public:
  bool bool_var;
  double double_var;
  double double_32bit;
  double double_16bit;
  float float_var;
  float float_16bit;
  int8_t i8_var;
  int16_t i16_var;
  int32_t i32_var;
  int64_t i64_var;
  uint8_t u8_var;
  uint16_t u16_var;
  uint32_t u32_var;
  uint64_t u64_var;
  uint8_t u8_all_bit_1;
  uint8_t u8_all_bit_2;
  uint8_t u8_1_bit;
  uint8_t u8_4_bit;
  uint8_t u8_array_1[128];
  uint8_t u8_array_2[64];

  bool EQ(const testing_full_var & data, float kp)
  {
    if (bool_var != data.bool_var ||
      double_var != data.double_var ||
      static_cast<int32_t>(double_32bit / kp) != static_cast<int32_t>(data.double_32bit / kp) ||
      static_cast<int16_t>(double_16bit / kp) != static_cast<int16_t>(data.double_16bit / kp) ||
      float_var != data.float_var ||
      static_cast<int16_t>(float_16bit / kp) != static_cast<int16_t>(data.float_16bit / kp) ||
      i8_var != data.i8_var ||
      i16_var != data.i16_var ||
      i32_var != data.i32_var ||
      i64_var != data.i64_var ||
      u8_var != data.u8_var ||
      u16_var != data.u16_var ||
      u32_var != data.u32_var ||
      u64_var != data.u64_var ||
      u8_all_bit_1 != data.u8_all_bit_1 ||
      u8_all_bit_2 != data.u8_all_bit_2 ||
      ((u8_1_bit & 0b1) != (data.u8_1_bit & 0b1)) ||
      ((u8_4_bit & 0xF) != (data.u8_4_bit & 0xF))) {return false;}
    for (uint a = 0; a < sizeof(u8_array_1); a++) {
      if (u8_array_1[a] != data.u8_array_1[a]) {return false;}
    }
    for (uint a = 0; a < sizeof(u8_array_2); a++) {
      if (u8_array_2[a] != data.u8_array_2[a]) {return false;}
    }
    return true;
  }

  void operator=(const testing_full_var & data)
  {
    bool_var = data.bool_var;
    double_var = data.double_var;
    double_32bit = data.double_32bit;
    double_16bit = data.double_16bit;
    float_var = data.float_var;
    float_16bit = data.float_16bit;
    i8_var = data.i8_var;
    i16_var = data.i16_var;
    i32_var = data.i32_var;
    i64_var = data.i64_var;
    u8_var = data.u8_var;
    u16_var = data.u16_var;
    u32_var = data.u32_var;
    u64_var = data.u64_var;
    u8_all_bit_1 = data.u8_all_bit_1;
    u8_all_bit_2 = data.u8_all_bit_2;
    u8_1_bit = data.u8_1_bit;
    u8_4_bit = data.u8_4_bit;
    for (uint a = 0; a < sizeof(u8_array_1); a++) {
      u8_array_1[a] = data.u8_array_1[a];
    }
    for (uint a = 0; a < sizeof(u8_array_2); a++) {
      u8_array_2[a] = data.u8_array_2[a];
    }
  }

  void init_type_1()
  {
    bool_var = true;
    double_var = -1234.5678;
    double_32bit = -34.789;
    double_16bit = 4.567;
    float_var = 78.914;
    float_16bit = -1.562;
    i8_var = -32;
    i16_var = -6543;
    i32_var = -654987;
    i64_var = 1234567;
    u8_var = 154;
    u16_var = 0xF3EA;
    u32_var = 0xF545'EACB;
    u64_var = 0xF51F'3233'1234'ABCD;
    u8_all_bit_1 = 0x51;
    u8_all_bit_2 = 0b1011'0011;
    u8_1_bit = 0xFF;
    u8_4_bit = 0b1111'1011;
    for (uint a = 0; a < sizeof(u8_array_1); a++) {
      u8_array_1[a] = 2 * a;
    }
    for (uint a = 0; a < sizeof(u8_array_2); a++) {
      u8_array_2[a] = 4 * a;
    }
  }

  void init_type_2()
  {
    bool_var = true;
    double_var = 1234.5678;
    double_32bit = 34.789;
    double_16bit = -4.567;
    float_var = -78.914;
    float_16bit = 1.562;
    i8_var = 32;
    i16_var = 6543;
    i32_var = 654987;
    i64_var = -1234567;
    u8_var = 102;
    u16_var = 0xABCD;
    u32_var = 0xEFEF'ABAB;
    u64_var = 0x1212'3232'5656'ABAB;
    u8_all_bit_1 = 0xA1;
    u8_all_bit_2 = 0b1011'1011;
    u8_1_bit = 0xFC;
    u8_4_bit = 0b1001'1101;
    for (uint a = 0; a < sizeof(u8_array_1); a++) {
      u8_array_1[a] = 1 * a;
    }
    for (uint a = 0; a < sizeof(u8_array_2); a++) {
      u8_array_2[a] = 3 * a;
    }
  }
};

std::shared_ptr<testing_full_var> callback_data = nullptr;
void callback(std::shared_ptr<testing_full_var> data)
{
  callback_data = data;
}

std::shared_ptr<EVM::Protocol<testing_full_var>> CreatDevice(
  std::string path,
  bool make_error = false,
  bool for_send = false)
{
  auto p = std::make_shared<EVM::Protocol<testing_full_var>>(path, for_send);
  if (!make_error) {p->LINK_VAR(p->GetData()->bool_var);}
  p->LINK_VAR(p->GetData()->double_var);
  p->LINK_VAR(p->GetData()->double_32bit);
  p->LINK_VAR(p->GetData()->double_16bit);
  p->LINK_VAR(p->GetData()->float_var);
  p->LINK_VAR(p->GetData()->float_16bit);
  p->LINK_VAR(p->GetData()->i8_var);
  p->LINK_VAR(p->GetData()->i16_var);
  p->LINK_VAR(p->GetData()->i32_var);
  p->LINK_VAR(p->GetData()->i64_var);
  p->LINK_VAR(p->GetData()->u8_var);
  p->LINK_VAR(p->GetData()->u16_var);
  p->LINK_VAR(p->GetData()->u32_var);
  p->LINK_VAR(p->GetData()->u64_var);
  p->LINK_VAR(p->GetData()->u8_all_bit_1);
  p->LINK_VAR(p->GetData()->u8_all_bit_2);
  p->LINK_VAR(p->GetData()->u8_1_bit);
  p->LINK_VAR(p->GetData()->u8_4_bit);
  p->LINK_VAR(p->GetData()->u8_array_1);
  p->LINK_VAR(p->GetData()->u8_array_2);
  p->SetDataCallback(callback);
  return p;
}

TEST(CommonProtocolTest_CAN, StateCollector) {
  EVM::StateCollector clct;
  clct.LogState(0);
  clct.LogState(234);
  auto clct_1 = clct.CreatChild();
  clct_1->LogState(1);
  auto clct_2 = clct.CreatChild();
  clct_2->LogState(2);
  auto clct_3 = clct.CreatChild();
  clct_3->LogState(3);

  auto clct_11 = clct_1->CreatChild();
  clct_11->LogState(1);
  clct_11->LogState(11);
  auto clct_12 = clct_1->CreatChild();
  clct_12->LogState(1);
  clct_12->LogState(12);

  auto clct_21 = clct_2->CreatChild();
  clct_21->LogState(2);
  clct_21->LogState(21);

  auto clct_22 = clct_2->CreatChild();
  clct_22->LogState(2);
  clct_22->LogState(22);

  clct.PrintfAllStateStr();
  ASSERT_EQ(CLCT(0), 1U);
  ASSERT_EQ(CLCT(1), 3U);
  ASSERT_EQ(CLCT(2), 3U);
  ASSERT_EQ(CLCT(3), 1U);
  ASSERT_EQ(CLCT(11), 1U);
  ASSERT_EQ(CLCT(12), 1U);
  ASSERT_EQ(CLCT(21), 1U);
  ASSERT_EQ(CLCT(22), 1U);
  ASSERT_EQ(CLCT(234), 1U);
}

// Testing normal usage STD_CAN with std_frame
TEST(CommonProtocolTest_CAN, initTest_success_0) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_success_0.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  ASSERT_FALSE(dv->IsRxTimeout());
  ASSERT_FALSE(dv->IsTxTimeout());
  ASSERT_FALSE(dv->IsRxError());
  ASSERT_TRUE(dv->Operate("start", std::vector<uint8_t>{0x1F, 0x5F}));
  ASSERT_TRUE(dv->Operate("close"));

  testing_full_var test_var;
  test_var.init_type_1();
  *dv->GetData() = test_var;
  ASSERT_EQ(callback_data, nullptr);
  ASSERT_TRUE(dv->SendSelfData());
  ASSERT_NE(callback_data, nullptr);
  ASSERT_TRUE(test_var.EQ(*callback_data, 0.01));
  callback_data = nullptr;

  test_var.init_type_2();
  *dv->GetData() = test_var;
  ASSERT_EQ(callback_data, nullptr);
  ASSERT_TRUE(dv->SendSelfData());
  ASSERT_NE(callback_data, nullptr);
  ASSERT_TRUE(test_var.EQ(*callback_data, 0.01));
  callback_data = nullptr;

  clct.PrintfAllStateStr();
  ASSERT_EQ(CLCT(), 0U);
}

// Testing normal usage FD_CAN with extended_frame
TEST(CommonProtocolTest_CAN, initTest_success_1) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_success_1.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  ASSERT_FALSE(dv->IsRxTimeout());
  ASSERT_FALSE(dv->IsTxTimeout());
  ASSERT_FALSE(dv->IsRxError());
  ASSERT_TRUE(dv->Operate("start", std::vector<uint8_t>{0x1F, 0x5F}));
  ASSERT_TRUE(dv->Operate("close"));

  testing_full_var test_var;
  test_var.init_type_1();
  *dv->GetData() = test_var;
  ASSERT_EQ(callback_data, nullptr);
  ASSERT_TRUE(dv->SendSelfData());
  ASSERT_NE(callback_data, nullptr);
  ASSERT_TRUE(test_var.EQ(*callback_data, 0.01));
  callback_data = nullptr;

  test_var.init_type_2();
  *dv->GetData() = test_var;
  ASSERT_EQ(callback_data, nullptr);
  ASSERT_TRUE(dv->SendSelfData());
  ASSERT_NE(callback_data, nullptr);
  ASSERT_TRUE(test_var.EQ(*callback_data, 0.01));
  callback_data = nullptr;

  clct.PrintfAllStateStr();
  ASSERT_EQ(CLCT(), 0U);
}

// Testing missing toml file
TEST(CommonProtocolTest_CAN, initTest_failed_0) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_0.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  clct.PrintfAllStateStr();
  ASSERT_GT(CLCT(EVM::ErrorCode::INIT_ERROR), 0U);
}

// Testing error protocol & missing common params
TEST(CommonProtocolTest_CAN, initTest_failed_1) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_1.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  ASSERT_FALSE(dv->Operate("test"));
  ASSERT_FALSE(dv->SendSelfData());
  ASSERT_TRUE(dv->IsRxTimeout());
  ASSERT_TRUE(dv->IsTxTimeout());
  ASSERT_TRUE(dv->IsRxError());

  clct.PrintfAllStateStr();
  ASSERT_GT(CLCT(EVM::ErrorCode::ILLEGAL_PROTOCOL), 0U);
}

// Testing missing key
TEST(CommonProtocolTest_CAN, initTest_failed_2) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_2.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  clct.PrintfAllStateStr();
  ASSERT_GE(CLCT(EVM::ErrorCode::INIT_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::CAN_ID_OUTOFRANGE), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::TOML_NOKEY_ERROR), 9U);
  ASSERT_GE(CLCT(EVM::ErrorCode::HEXTOUINT_ILLEGAL_START), 2U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_SIZE), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEVAR_ILLEGAL_VARTYPE), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEVAR_ILLEGAL_VARNAME), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEARRAY_ILLEGAL_ARRAYNAME), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE), 2U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULECMD_ILLEGAL_CMDNAME), 1U);
}

// Testing var error
TEST(CommonProtocolTest_CAN, initTest_failed_3) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_3.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  clct.PrintfAllStateStr();
  ASSERT_GE(CLCT(EVM::ErrorCode::INIT_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::DATA_AREA_CONFLICT), 4U);
  ASSERT_GE(CLCT(EVM::ErrorCode::CAN_ID_OUTOFRANGE), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::TOML_TYPE_ERROR), 3U);
  ASSERT_GE(CLCT(EVM::ErrorCode::HEXTOUINT_ILLEGAL_CHAR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::HEXTOUINT_ILLEGAL_START), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULE_SAMENAME_ERROR), 2U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEVAR_ILLEGAL_PARSERTYPE), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_SIZE), 9U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEVAR_ILLEGAL_PARSERPARAM_VALUE), 11U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEVAR_ILLEGAL_VARTYPE), 1U);
}

// Testing array error
TEST(CommonProtocolTest_CAN, initTest_failed_4) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_4.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  clct.PrintfAllStateStr();
  ASSERT_GE(CLCT(EVM::ErrorCode::INIT_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::DATA_AREA_CONFLICT), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::TOML_TYPE_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULE_SAMENAME_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEARRAY_SAMECANID_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE), 8U);
}

// Testing CMD error
TEST(CommonProtocolTest_CAN, initTest_failed_5) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_5.toml";
  auto dv = CreatDevice(path);
  auto & clct = dv->GetErrorCollector();

  clct.PrintfAllStateStr();
  ASSERT_GE(CLCT(EVM::ErrorCode::INIT_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULECMD_CTRLDATA_ERROR), 4U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULECMD_SAMECMD_ERROR), 1U);
}

class test_1
{
public:
  struct flag_
  {
    int test = 0;
  } flag;
};

// Testing src code error
TEST(CommonProtocolTest_CAN, initTest_failed_6) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_6.toml";
  auto dv = CreatDevice(path, true);
  auto & clct = dv->GetErrorCollector();
  int test = 0;
  test_1 test_1_;

  dv->LINK_VAR(dv->GetData()->double_32bit);
  dv->LINK_VAR(test);
  dv->LINK_VAR((test_1_).flag.test);
  ASSERT_TRUE(dv->Operate("CMD_0"));
  ASSERT_TRUE(dv->Operate("CMD_1"));
  ASSERT_FALSE(dv->Operate("CMD_1", std::vector<uint8_t>(8)));
  ASSERT_FALSE(dv->Operate("CMD_2"));
  ASSERT_FALSE(dv->SendSelfData());

  clct.PrintfAllStateStr();
  ASSERT_GE(CLCT(EVM::ErrorCode::INIT_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::DATA_AREA_CONFLICT), 2U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULEARRAY_ILLEGAL_PARSERPARAM_VALUE), 21U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RULECMD_MISSING_ERROR), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::DOUBLE_SIMPLIFY_ERROR), 2U);
  ASSERT_GE(CLCT(EVM::ErrorCode::FLOAT_SIMPLIFY_ERROR), 4U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RUNTIME_UNEXPECT_ORDERPACKAGE), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RUNTIME_SIZEOVERFLOW), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RUNTIME_SIZENOTMATCH), 1U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RUNTIME_OPERATE_ERROR), 2U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RUNTIME_NOLINK_ERROR), 7U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RUNTIME_SAMELINK_ERROR), 2U);
  ASSERT_GE(CLCT(EVM::ErrorCode::RUNTIME_ILLEGAL_LINKVAR), 2U);
}

class testing_var
{
public:
  uint64_t u64_var;
  uint8_t u8_array[8];
};

std::shared_ptr<testing_var> callback_data_testing_var = nullptr;
void testing_var_callback(std::shared_ptr<testing_var> data)
{
  callback_data_testing_var = data;
}

// Testing Operate data get
TEST(CommonProtocolTest_CAN, initTest_failed_7) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_failed_7.toml";
  auto dv = EVM::Protocol<testing_var>(path);
  dv.LINK_VAR(dv.GetData()->u64_var);
  dv.LINK_VAR(dv.GetData()->u8_array);
  dv.SetDataCallback(testing_var_callback);

  ASSERT_EQ(callback_data_testing_var, nullptr);
  dv.Operate(
    "CMD_0", std::vector<uint8_t> {
    0xA3, 0xA4, 0xA5, 0xA6, 0xA7, 0xA8
  });
  dv.Operate(
    "CMD_1", std::vector<uint8_t> {
    0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8
  });
  ASSERT_NE(callback_data_testing_var, nullptr);
  ASSERT_EQ(callback_data_testing_var->u64_var, 0xA1A2A3A4A5A6A7A8U);
  for (int a = 0; a < 8; a++) {
    ASSERT_EQ(callback_data_testing_var->u8_array[a], 0xB1U + a);
  }
  callback_data_testing_var = nullptr;

  ASSERT_EQ(callback_data_testing_var, nullptr);
  dv.Operate(
    "CMD_0", std::vector<uint8_t> {
    0xB3, 0xB4, 0xB5, 0xB6, 0xB7, 0xB8
  });
  dv.Operate(
    "CMD_2", std::vector<uint8_t> {
    0xC3, 0xC4, 0xC5, 0xC6, 0xC7, 0xC8
  });
  ASSERT_NE(callback_data_testing_var, nullptr);
  ASSERT_NE(callback_data_testing_var->u64_var, 0xA1A2A3A4A5A6A7A8U);
  for (int a = 0; a < 8; a++) {
    ASSERT_EQ(callback_data_testing_var->u8_array[a], 0xC1U + a);
  }
  callback_data_testing_var = nullptr;
}

// Testing normal usage STD_CAN with std_frame send error
TEST(CommonProtocolTest_CAN, initTest_failed_8) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_success_0.toml";
  auto dv = CreatDevice(path, false, true);
  auto & clct = dv->GetErrorCollector();

  ASSERT_FALSE(dv->SendSelfData());
  ASSERT_FALSE(dv->Operate("start"));

  clct.PrintfAllStateStr();
  ASSERT_GE(CLCT(EVM::ErrorCode::CAN_STD_SEND_ERROR), 34U);
}

// Testing normal usage FD_CAN with extended_frame send error
TEST(CommonProtocolTest_CAN, initTest_failed_9) {
  std::string path = std::string(PASER_PATH) + "/can/initTest_success_1.toml";
  auto dv = CreatDevice(path, false, true);
  auto & clct = dv->GetErrorCollector();

  ASSERT_FALSE(dv->SendSelfData());
  ASSERT_FALSE(dv->Operate("start"));

  clct.PrintfAllStateStr();
  ASSERT_GE(CLCT(EVM::ErrorCode::CAN_FD_SEND_ERROR), 14U);
}

int main(int argc, char ** argv)
{
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
