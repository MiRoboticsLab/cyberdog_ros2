# COMMON_PROTOCOL
[中文版本](README.md)

Common_protocol is a general peripheral abstract class, which can dynamically and flexibly configure the data analysis method of peripherals based on a certain communication protocol, and all communication protocols are abstracted into a description file, which can be loaded in the form of external loading. Make changes without recompiling the software source code,<u>In order to prevent errors in the modification, each description file is preset in the program in the initial compilation state, and the internal preset file can be automatically loaded when the external file is missing or an error occurs.(TBD)</u>

The example simplifies the method of using peripheral devices by loading different description files to be compatible with different communication transmission protocols and formats. Now only need to care about the use of messages and the issuance of instruction data, from complex message encoding, decoding, sending and It is liberated from receiving, and different communication transmission protocols also have a unified common interface

***
## 1.Structure introduction

The entire feature pack uses namespace `common_protocol`

The function directory structure is as follows:
```
include
├── common_protocol
│   ├── can_protocol.hpp
│   ├── common_protocol.hpp
│   ├── common.hpp
│   └── protocol_base.hpp
└── common_parser
    └── can_parser.hpp
```
- common_protocol : General equipment, used to store the main body code
    - common.hpp : General and tool codes
    - common_protocol.hpp : Unified external interface
    - protocol_base.hpp : Base class interface of different protocols
    - [accomplish] can_protocol.hpp : Function realization of CAN protocol transmission, derived from protocol_base
- common_parser : Universal parser, used to store protocol parsing code
    - [accomplish] can_parser.hpp : Analysis and realization of CAN protocol transmission

The description file storage directory see : [`cyberdog_bridges/README.md`](TBD)

### Class Protocol

This class is used for the main external interface:

```cpp
#define XNAME(x) (#x)
#define LINK_VAR(var) LinkVar( \
    XNAME(var), \
    cyberdog::common::ProtocolData(sizeof((var)), static_cast<void *>(&(var))))

namespace common
{
template<typename TDataClass>
class Protocol
{
public:
  explicit Protocol(const std::string & protocol_toml_path);

  std::shared_ptr<TDataClass> GetData();

  // please use "#define LINK_VAR(var)" instead
  void LinkVar(const std::string origin_name, const ProtocolData & var);

  bool Operate(const std::string & CMD, const std::vector<uint8_t> & data = std::vector<uint8_t>());

  bool SendSelfData();

  void SetDataCallback(std::function<void(std::shared_ptr<TDataClass>)> callback);

  StateCollector & GetErrorCollector();
};  // class Protocol
}  // namespace common
```

> Constructor, create instance object through external description file
> ```cpp
> explicit Protocol(const std::string & protocol_toml_path, bool for_send = false);
> ```
> - `protocol_toml_path` : Description file address
> - `for_send` : Send via toml description file, or receive via toml description file (receive by default)

> Get custom TDataClass type data: return shared_ptr pointing to internal data
> ```cpp
> std::shared_ptr<TDataClass> GetData();
> ```
> Note : Generally used for data filling of `LINK_DATA` and `for_send` devices, please use the callback function method for receiving

> Link TDataClass data so that variables can be parsed into TDataClass through various transmission protocols
> ```cpp
> void LinkVar(const std::string origin_name, const ProtocolData & var);
> ```
> Note : Generally use the macro `LINK_VAR(var)` instead, see the example below for detailed usage

> Device operation function, issue instructions and data: return whether the complete transmission is successful
> ```cpp
> bool Operate(const std::string & CMD, const std::vector<uint8_t> & data = std::vector<uint8_t>());
> ```
> - `CMD` : Command string (specified in the description file)
> - `data` : The data carried by the instruction

> Send the internal data modified by `GetData()` through the parsing protocol: return whether the complete transmission is successful
> ```cpp
> bool SendSelfData();
> ```
> Note : If the device is not in the sending mode (ie `for_send=false`), the device can also send, but it is not recommended to use it in non-testing

> Set the receive callback function
> ```cpp
> void SetDataCallback(std::function<void(std::shared_ptr<TDataClass>)> callback);
> ```
> - `callback` : Set callback function

> Get whether the receiving method has timed out: return whether it has timed out
> ```cpp
> bool IsRxTimeout()
> ```
> Note : The timeout record will be cleared after successful reception

> Get whether the sending method has timed out: return whether it has timed out
> ```cpp
> bool IsTxTimeout()
> ```
> Note : The timeout record will be cleared when it is successfully sent

> Get whether there is an error in the receiving mode: Return whether there is an error
> ```cpp
> bool IsRxError()
> ```
> Note : When the data in the description file is received successfully and completely, the error record will be cleared

> Get error collector: return is the collector reference
> ```cpp
> StateCollector & GetErrorCollector()
> ```
> Note : The specific error code is in `common.hpp`

Usage example:
```cpp
class Acc
{
public:
  float x;
  float y;
  float z;
};

void callback(std::shared_ptr<Acc> data)
{
  printf("callback, Acc.x=%f, Acc.y=%f, Acc.z=%f\n", data->x, data->y, data->z);
}

int main(int argc, char ** argv)
{
  UNUSED_VAR(argc);
  UNUSED_VAR(argv);

  // receive-operate mode
  cyberdog::common::Protocol<Acc> protocol_1("parser/can/acc_protocol/acc_1.toml");
  protocol_1.LINK_VAR(protocol_1.GetData()->x);
  protocol_1.LINK_VAR(protocol_1.GetData()->y);
  protocol_1.LINK_VAR(protocol_1.GetData()->z);
  protocol_1.SetDataCallback(callback);

  std::vector<uint8_t> data = std::vector<uint8_t>(6);
  protocol_1.Operate("start", data);
  
  // for_send mode
  cyberdog::common::Protocol<Acc> protocol_2("parser/can/acc_protocol/acc_2.toml");
  protocol_2.LINK_VAR(protocol_2.GetData()->x);
  protocol_2.LINK_VAR(protocol_2.GetData()->y);
  protocol_2.LINK_VAR(protocol_2.GetData()->z);

  std::shared_ptr acc_data = protocol_2.GetData();
  acc_data->x = 1.2;
  acc_data->y = 3.4;
  acc_data->z = 5.6;
  protocol_2.SendSelfData();

  return 0;
}
```

***
## 2.Communication description file

### CAN communication description file usage

Example:

```toml
# -- common params -- #
# protocol = "can"             (string)
# name = "can_protocol_name"     (string)
# [delete] for_send = false  (true / false)
protocol = "can"
name = "can_protocol_1"

# -- can params -- #
# can_interface = "can0"             (string)
# [optional] extended_frame = false  (true / false)
# [optional] canfd_enable = false    (true / false)
# [optional] timeout_us = 3'000'000  (int64)
can_interface = "can0"
extended_frame = true
canfd_enable = false
timeout_us = 1000000

# -- data_var -- #
# [[var]]
# can_id = "0x0300 00 00" / "0x0300'00'00"  (string-HEX)
# var_name = "var_name_1"                   (string)
# var_type = "float"                      
#            (float / double / i64 / i32 / i16 / i8 / u64 / u32 / u16 / u8 / bool)
# parser_param = [0, 3]                     (array<uint8>[2] / array<uint8>[3])
# [optional] var_zoom = 1.0                 (float)
# [optional] parser_type = "auto"           (bit / var / auto)
# [optional] description = ""               (string)

[[var]]
can_id = "0x0300 00 00"
var_name = "example_var_1"
var_type = "double"
parser_param = [0, 7]
var_zoom = 0.0001
description = "this is example named example_var_1"

[[var]]
can_id = "0x0300'00'01"
var_name = "example_var_2"
var_type = "uint8"
parser_param = [0, 7, 4]
parser_type = "bit"
description = "this is example named example_var_2"

# -- data_array -- #
# [[array]]
# can_package_num = 8          (size_t)
# can_id = ["0x200", "0x207"]  (array<string-HEX>[2] / array<string-HEX>[can_package_num])
# array_name = "array_name_1"  (string)
# [optional] description = ""  (string)

[[array]]
can_package_num = 8
can_id = ["0x200", "0x207"]
array_name = "example_array_1"
description = "this is example named example_array_1"

[[array]]
can_package_num = 4
can_id = ["0x200", "0x201", "0x202", "0x203"]
array_name = "example_array_2"
description = "this is example named example_array_2"

# -- cmd -- #
# [[cmd]]
# cmd_name = "cmd_name_1"      (string)
# can_id = "0x02"              (string-HEX)
# [optional] ctrl_len = 0      (uint8)
# [optional] ctrl_data = []    (array<string-HEX>[x] : where x <= ctrl_len)
# [optional] description = ""  (string)

[[cmd]]
cmd_name = "example_cmd_1"
can_id = "0x02"
ctrl_len = 2
ctrl_data = ["0x06", "0x13"]
description = "this is example named example_cmd_1"
```

Parsing:
- `common params` : General parameters
    - `protocol`: the communication protocol specified by the description file
    - `name`: the name of the device `common_protocol`, this value will be used to indicate in the log
    - ~~[Delete] `for_send`: The device receives or sends from `TDataClass`, if it is `false`, the receiving thread is turned on to receive data from the `CAN bus` to `TDataClass`; if it is `true`, the receiving thread is closed , Use the `SendSelfData()` function to send the `TDataClass` to the `CAN bus`~~ (instead of passing in the class common_protocol, in order to achieve the same description file for sending and receiving)
- `can params`: CAN protocol parameters
    - `can_interface`: The CAN bus used, generally `"can0"`, `"can1"` or `"can2"`
    - [Optional] `extended_frame`: Whether to use extended frame (mainly for sending and receiving are fully compatible), the default default value is: `false`
    - [Optional] `canfd_enable`: Whether to use CAN_FD (requires system settings and CAN transceiver hardware support), the default default value is: `false`
    - [Optional] `timeout_us`: Receiving or sending timeout time (microseconds), mainly used to judge the receiving disconnection and prevent the card from being unable to proceed in the receiving or sending function during destructuring, the effective value is `1'000(1ms) )` to `3'000'000(3s)`, default default value: `3'000'000(3s)`
- `data_var`: CAN protocol variable analysis rules
    - `can_id`: ID to be received, a `hexadecimal string` beginning with `"0x"` or `"0X"`, the symbols `"'" (single quotation mark)` and `“ ”( Space)` Make arbitrary divisions to facilitate reading and writing, such as: `0x1FF'12'34` or `0x123 45 67`
    - `var_name`: The name of the variable that needs to be resolved (that is, the name of the variable linked with `LINK_VAR(var)` in the code)
    - `var_type`:
        - The variable type that needs to be resolved (that is, the variable type linked by `LINK_VAR(var)` in the code), and different types can be used in special circumstances (such as parsing `u8` to `u16` in the code), But the size of `size_of` cannot exceed the size in the code (that is, `u64` cannot be parsed into `u8` in the code)
        - Support parse format: `float` / `double` / `i64` / `i32` / `i16` / `i8` / `u64` / `u32` / `u16` / `u8` / `bool`
    - `parser_param`: parse parameters, there will be different lengths according to different parse types
        - Parse as a variable (ie `parser_type = "var"`):
            - `parser_param` is `array<uint8>[2]`, which is a `uint8` array of length 2 and satisfies `0 <= parser_param[0] <= parser_param[1] < MAX_CAN_LEN`, where `MAX_CAN_LEN` is in 8 in `STD_CAN`, 64 in `FD_CAN`
            - Priority during parsing <u>High order priority in binary</u> The parsing principle combines `can_data[parser_param[0]]` to `can_data[parser_param[1]]` in binary form into one variable
            - `float` and `double` are first merged in the analytical form of <u>high-order first in binary</u> under special circumstances, and then generated into a variable in the analytical form of <u>decimal point precision scaling</u>
            - See the example table below for detailed analysis (`example 1` to `example 3`)
        - Parse in bit form (ie `parser_type = "bit"`):
            - `parser_param` is `array<uint8>[3]`, which is an array of `uint8` with a length of 3, and satisfies `0 <= parser_param[0] < MAX_CAN_LEN && 8 > parser_param[1] >= parser_param[ 2] >= 0`, where `MAX_CAN_LEN` is 8 in `STD_CAN` and is 64 in `FD_CAN`
            - During parsing, the data in `can_data[parser_param[0]]` mid-high `parser_param[1]` bit to low `parser_param[2]` bit is taken out and combined by the <u>shift and bit-and</u> parsing principle Shift `parser_param[2]` bits to the right
            - See the example table below for detailed analysis (`Example 4`)
    - [Optional] `var_zoom`: zoom parameter, combined with the variable in the form of multiplication, only `float` and `double` are valid, the default value is: `1.0`
    - [Optional] `parser_type`:
        - The type of parsing can be specified manually or inferred automatically based on `parser_param.length()`
        - Support analysis type: `var` / `bit` / `auto (default default value)`
    - [Optional] `description`: Notes and usage description
- `data_array`: CAN protocol array parsing rules
    - `can_package_num`: The expected number of CAN data frames received
    - `can_id`: Expected to receive CAN_ID array as array value
        - Manually specify all CAN_IDs (i.e. `can_id.length() == can_package_num`):
            - `can_id` is `array<string-HEX>[can_package_num]`, which is a `hexadecimal string` with length `can_package_num`
            - All data will be filled in the order of the array `index` according to the specified CAN_ID order, that is, the `index` in `can_id` can be queried by CAN_ID, and the filling will be carried out based on `array[index * 8]`
            - See the example table below for detailed analysis (`Example 5`)
        - Automatically specify continuous CAN_ID (ie `can_id.length() != can_package_num && can_id.length() == 2 && can_id[0] < can_id[1]`)
            - `can_id` is `array<string-HEX>[2]`, which is a `hexadecimal string` of length 2
            - All data will be filled in the order of the array `index` according to the specified CAN_ID order, that is, the `index` in `can_id` can be queried by CAN_ID, and the filling will be carried out based on `array[index * 8]`
            - See the example table below for detailed analysis (`Example 6`)
    - `array_name`: the name of the array to be parsed (that is, the name of the array linked with `LINK_VAR(var)` in the code)
    - [Optional] `description`: Notes and usage description
- `cmd`: CAN protocol issued command parsing rules (`Example 7`)
    - `cmd_name`: command name (i.e. the command used in the code to operate with the `Operate()` function)
    - `can_id`: CAN_ID for command transmission
    - [Optional] `ctrl_len`: control segment length, which must meet `0 <= ctrl_len < MAX_CAN_LEN`, where `MAX_CAN_LEN` is 8 in `STD_CAN` and 64 in `FD_CAN`, the default value is: ` 0`
    - [Optional] `ctrl_data`:
        - The control segment data is `hexadecimal string array` (the hexadecimal string does not exceed the two digits of hexadecimal (ie u8)), which satisfies `0 <= ctrl_data.length() <= ctrl_len `, the default default value is: `[]`
    - [Optional] `description`: Notes and usage description

Example set:

> Example 1:
>
> | Analysis Rules | Data |
> | :----: | :----: |
> | var_name = "example_1" | can_data[0] = 0x12 |
> | var_type = "u16"       | can_data[1] = 0x34 |
> | parser_param = [1, 2]  | can_data[2] = 0x56 |
>
> The equivalent code after parsing is:
> ```cpp
> uint16 example_1 = 0x3456;
> ```
>
> This example shows that the parameter `[1, 2]`, that is, the `can_data` array `index` is `1` and `2`, a total of 2 `u8`<u>high-order priority in binary</u> The process of merging the analytic form into the variable `example_1` of type `u16`

> Example 2:
> 
> | Analysis Rules | Data |
> | :----: | :----: |
> | var_name = "example_2" | can_data[0] = 0x12 |
> | var_type = "float"     | can_data[1] = 0x34 |
> | parser_param = [0, 3]  | can_data[2] = 0x56 |
> | var_zoom = 0.0001      | can_data[3] = 0x78 |
> 
> The equivalent code after parsing is:
> ```cpp
> uint32 tmp_2 = 0x12345678;
> float example_2 = *((float *)(&tmp_2)) * 0.0001;
> ```
>
> This example shows the parameter `[0, 3]`, that is, the `can_data` array `index` is `0`, `1`, `2`, and `3`, a total of 4 `u8`<u> Prioritize the process of merging into a `float` type variable `example_2` according to the analytic form of binary</u>

> Example 3:
> 
> | Analysis Rules | Data |
> | :----: | :----: |
> | var_name = "example_3" | can_data[0] = 0x12 |
> | var_type = "double"    | can_data[1] = 0x34 |
> | parser_param = [0, 3]  | can_data[2] = 0x56 |
> | var_zoom = 0.0001      | can_data[3] = 0x78 |
> 
> The equivalent code after parsing is:
> ```cpp
> uint32 tmp_2 = 0x12345678;
> double example_2 = (double)(tmp_2) * 0.0001;
> ```
>
> This example shows using the parameter `[0, 3]`, that is, the `can_data` array `index` is `0`, `1`, `2`, and `3`, a total of 4 `u8` first <u> The high order first is merged into `u32` according to the analytical form of binary</u>, and then the process of scaling the decimal point to obtain the `double` type variable `example_3` in the analytical form of <u>decimal point precision scaling</u>
> ***
> __Note__: When `var_type` is `float` or `double`, when the number of parsing `u8` specified by `parser_param` is not equal to `size` of `flaot` and `double` and is `size` The even-numbered one (1/2, 1/4) of the, will use the <u>high-order priority to combine according to the binary</u>, and then use the analytical form of <u>decimal point precision scaling</u>:
>-That is, the `size` of `float` is 4 `u8`, when the number of parsing `u8` specified by `parser_param` is 2
>-That is, the `size` of `double` is 8 `u8`, when the number of parsing `u8` specified by `parser_param` is 4 or 2

> Example 4:
> 
> | Analysis Rules | Data |
> | :----: | :----: |
> | var_name = "example_4"   | can_data[0] = 0x12 (0b0001'0010) |
> | var_type = "u8"          | can_data[1] = 0x34 (0b00<u>11</u>'0100) |
> | parser_param = [1, 5, 4] | can_data[2] = 0x56 (0b0101'0110) |
> 
> The equivalent code after parsing is:
> ```cpp
> uint8 tmp_4 = 0x34;  // 0b0011'0100
> uint8 example_4 = (uint8)((0b0011'0100 & 0b0011'0000) >> 4);
> uint8 example_4 = 0b0000'0011;
> ```
>
> This example shows the use of the parameter `[1, 5, 4]`, that is, the `can_data` array `index` is `1`, after taking the high `5` bits to the low `4` bits (&= 0b0011'0000) Shift right `4`

> Example 5:
> 
> | Analysis Rules | 0x200 | 0x201 | 0x2FF |
> | :----: | :----: | :----: | :----: |
> | can_package_num = 3                  | can_data[0] = 0xA0 | can_data[0] = 0xB0 | can_data[0] = 0xC0 |
> | can_id = ["0x201", "0x2FF", "0x200"] | can_data[1] = 0xA1 | can_data[1] = 0xB1 | can_data[1] = 0xC1 |
> | array_name = example_5               | can_data[2] = 0xA2 | can_data[2] = 0xB2 | can_data[2] = 0xC2 |
> |                                      | ... = 0xA3         | ... = 0xB3         | ... = 0xC3         |
> |                                      | ... = 0xA4         | ... = 0xB4         | ... = 0xC4         |
> |                                      | ... = 0xA5         | ... = 0xB5         | ... = 0xC5         |
> |                                      | ... = 0xA6         | ... = 0xB6         | ... = 0xC6         |
> |                                      | ... = 0xA7         | ... = 0xB7         | ... = 0xC7         |
> 
> The equivalent code after parsing is:
> ```cpp
> uint8 example_5[24] = {
>   0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
>   0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
>   0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7
> };
> ```
>
> This example shows that the parameters `["0x201", "0x2FF", "0x200"]` are parsed and filled into the array `example_5` in the order of `0x201`, `0x2FF`, and `0x200`

> Example 6:
> 
> | Analysis Rules | 0x200 | 0x201 | 0x202 |
> | :----: | :----: | :----: | :----: |
> | can_package_num = 3         | can_data[0] = 0xA0 | can_data[0] = 0xB0 | can_data[0] = 0xC0 |
> | can_id = ["0x200", "0x202"] | can_data[1] = 0xA1 | can_data[1] = 0xB1 | can_data[1] = 0xC1 |
> | array_name = example_6      | can_data[2] = 0xA2 | can_data[2] = 0xB2 | can_data[2] = 0xC2 |
> |                             | ... = 0xA3         | ... = 0xB3         | ... = 0xC3         |
> |                             | ... = 0xA4         | ... = 0xB4         | ... = 0xC4         |
> |                             | ... = 0xA5         | ... = 0xB5         | ... = 0xC5         |
> |                             | ... = 0xA6         | ... = 0xB6         | ... = 0xC6         |
> |                             | ... = 0xA7         | ... = 0xB7         | ... = 0xC7         |
> 
> The equivalent code after parsing is:
> ```cpp
> uint8 example_6[24] = {
>   0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
>   0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
>   0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7
> };
> ```
>
> This example shows the parameters `["0x200", "0x202"]`, which are parsed and filled into the array `example_6` in order from `0x200` to `0x202`

> Example 7:
>
> | Analysis Rules | Data |
> | :----: | :----: |
> | cmd_name = "cmd_1"           | cmd_data[0] = 0x12 |
> | can_id = "0x1F2"             | cmd_data[1] = 0x34 |
> | ctrl_len = 4                 | cmd_data[2] = 0x56 |
> | ctrl_data = ["0xA1", "0xA2"] | cmd_data[3] = 0x78 |
>
> How to use:
> ```cpp
> can_protocol_1.Operate("cmd_1", cmd_data);
> ```
>
> The equivalent code after use is:
> ```cpp
> struct can_frame tx_frame;
> tx_frame.can_id = 0x1F2;
> tx_frame.can_dlc = 8;
> tx_frame.data[0] = 0xA1;
> tx_frame.data[1] = 0xA2;
>
> tx_frame.data[4] = 0x12;
> tx_frame.data[5] = 0x34;
> tx_frame.data[6] = 0x56;
> tx_frame.data[7] = 0x78;
> can_send(tx_frame);
> ```
>
> This example shows using `cmd_1` to carry `cmd_data` to issue instructions, that is, according to `ctrl_len`, the first 4 u8s are all control section data and are filled with `ctrl_data`, but 2 u8s are left blank due to insufficient data , The remaining 4 u8 are filled in according to the incoming parameter `cmd_data`
