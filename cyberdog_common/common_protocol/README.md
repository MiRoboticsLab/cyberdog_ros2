# COMMON_PROTOCOL
[English Version](README_EN.md)

common_protocol是一个通用的外设抽象类，可动态灵活的配置基于某种通信协议外设的数据解析方式，且所有通信协议都被抽象为一种描述文件，该文件通过外部加载的形式载入，可在不重新编译软件源代码的情况下进行更改，<u>为防止改动出现错误，每一个描述文件都在初始编译态预置在了程序内部，当外部文件缺失或出现错误的情况下可自动载入内部预置文件(TBD)</u>

实例通过载入不同描述文件以兼容不同的通信传输协议以及格式，简化了使用外设设备的方法，现在只需要关心消息的使用与指令数据的下发，从复杂的消息编码、解码、发送和接收中解放了出来，不同的通信传输协议也有了统一的通用接口

***
## 1.结构简介

整个功能包使用命名空间`common_protocol`

功能目录结构如下:
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
- common_protocol : 通用设备，用于存放主体代码
    - common.hpp : 通用及工具代码
    - common_protocol.hpp : 对外统一接口
    - protocol_base.hpp : 不同协议的基类接口
    - [实现] can_protocol.hpp : CAN协议传输的功能实现，从protocol_base派生
- common_parser : 通用解析器，用于存放协议解析代码
    - [实现] can_parser.hpp : CAN协议传输的解析实现

描述文件存放目录见 : [`cyberdog_bridges/README.md`](TBD)规定

### Class Protocol

该类用于对外的主要接口:

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

> 构造函数，通过外部描述文件创建实例对象
> ```cpp
> explicit Protocol(const std::string & protocol_toml_path, bool for_send = false);
> ```
> - `protocol_toml_path` : 描述文件地址
> - `for_send` : 通过toml描述文件发送，或通过toml描述文件接收(默认为接收)

> 获取自定TDataClass类型数据 : 返回指向内部数据的shared_ptr
> ```cpp
> std::shared_ptr<TDataClass> GetData();
> ```
> Note : 一般用于`LINK_DATA`和`for_send`设备的数据装填，接收请使用回调函数方式

> 链接TDataClass数据，以便通过各种传输协议解析出变量到TDataClass中
> ```cpp
> void LinkVar(const std::string origin_name, const ProtocolData & var);
> ```
> Note : 一般使用宏`LINK_VAR(var)`代替，详细用法见下方example

> 设备操作函数，下发指令及数据 : 返回是否完整发送成功
> ```cpp
> bool Operate(const std::string & CMD, const std::vector<uint8_t> & data = std::vector<uint8_t>());
> ```
> - `CMD` : 指令字符串(描述文件中规定)
> - `data` : 指令所携带数据

> 将`GetData()`修改的内部数据通过解析协议发送出去 : 返回是否完整发送成功
> ```cpp
> bool SendSelfData();
> ```
> Note : 如设备不为发送模式(即`for_send=false`)，设备同样可以进行发送，但不建议在非测试时使用

> 设置接收回调函数
> ```cpp
> void SetDataCallback(std::function<void(std::shared_ptr<TDataClass>)> callback);
> ```
> - `callback` : 设置的回调函数

> 获取接收方式是否超时 : 返回是否超时
> ```cpp
> bool IsRxTimeout()
> ```
> Note : 当成功接收后将清空超时记录

> 获取发送方式是否超时 : 返回是否超时
> ```cpp
> bool IsTxTimeout()
> ```
> Note : 当成功发送后将清空超时记录

> 获取接收方式是否出错 : 返回是否出错
> ```cpp
> bool IsRxError()
> ```
> Note : 当成功完整接收描述文件内数据将清空出错记录

> 获取错误收集器 : 返回是搜集器引用
> ```cpp
> StateCollector & GetErrorCollector()
> ```
> Note : 具体错误代码在`common.hpp中`

使用示例:
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
## 2.通信描述文件

### CAN通信描述文件使用

示例:

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

解析:
- `common params` : 通用参数
    - `protocol` : 该描述文件所规定的通信协议
    - `name` : 该设备`common_protocol`的名称，在log中会使用该值来提示
    - ~~[删除] `for_send` : 该设备从`TDataClass`接收或发送，如为`false`则开启接收线程从`CAN总线`接收数据到`TDataClass`; 如为`true`则关闭接收线程，使用`SendSelfData()`函数将`TDataClass`发送到`CAN总线`~~(改为在class common_protocol中传入，以实现收发使用同一份描述文件)
- `can params` : CAN协议参数
    - `can_interface` : 使用的CAN总线，一般为`"can0"`, `"can1"` 或 `"can2"`
    - [可选] `extended_frame` : 是否使用扩展帧(主要针对发送，接收是全兼容的)，默认缺省值为 : `false`
    - [可选] `canfd_enable` : 是否使用CAN_FD(需要系统设置及CAN收发器硬件支持)，默认缺省值为 : `false`
    - [可选] `timeout_us` : 接收或发送超时时间(微秒)，主要用于判断接收掉线和防止析构时卡在接收或发送函数中无法进行，有效值为`1'000(1ms)`到`3'000'000(3s)`，默认缺省值 : `3'000'000(3s)`
- `data_var` : CAN协议变量解析规则
    - `can_id` : 需要接收的ID，以`"0x"`或`"0X"`开头的`十六进制字符串`，可使用符号`“’”(单引号)`和`“ ”(空格)`进行任意分割以方便阅读和书写，如 : `0x1FF'12'34` 或 `0x123 45 67` 
    - `var_name` : 需要解析到的变量名称(即在代码中使用`LINK_VAR(var)`链接的变量名称)
    - `var_type` :
        - 需要解析的变量类型(即在代码中使用`LINK_VAR(var)`链接的变量类型)，在及其特殊情况下可使用不同类型(如将`u8`解析到代码中的`u16`)，但`size_of`的大小不能超过代码中的大小(即不可将`u64`解析到代码中的`u8`)
        - 支持解析格式 : `float` / `double` / `i64` / `i32` / `i16` / `i8` / `u64` / `u32` / `u16` / `u8` / `bool`
    - `parser_param` : 解析参数，根据不同解析类型会存在不同长度
        - 以变量形式解析(即`parser_type = "var"`) :
            - `parser_param`为`array<uint8>[2]`，即长度为2的`uint8`数组，且满足`0 <= parser_param[0] <= parser_param[1] < MAX_CAN_LEN`，其中`MAX_CAN_LEN`在`STD_CAN`中为8，`FD_CAN`中为64
            - 解析时优先<u>以高位优先按二进制</u>解析原则将`can_data[parser_param[0]]`到`can_data[parser_param[1]]`以二进制形式合并为一个变量
            - `float`与`double`在特殊情况下先<u>以高位优先按二进制</u>的解析形式合并，再以<u>小数点精度缩放</u>的解析形式生成为一个变量
            - 详细解析见下方示例表格(`例1`到`例3`)
        - 以位形式解析(即`parser_type = "bit"`) :
            - `parser_param`为`array<uint8>[3]`，即长度为3的`uint8`数组，且满足`0 <= parser_param[0] < MAX_CAN_LEN && 8 > parser_param[1] >= parser_param[2] >= 0`，其中`MAX_CAN_LEN`在`STD_CAN`中为8，`FD_CAN`中为64
            - 解析时通过<u>移位和位与</u>解析原则将`can_data[parser_param[0]]`中高`parser_param[1]`位到低`parser_param[2]`位中的数据取出并右移`parser_param[2]`位
            - 详细解析见下方示例表格(`例4`)
    - [可选] `var_zoom` : 缩放参数，以乘法形式与变量合并，仅`float`和`double`有效，默认缺省值为 : `1.0`
    - [可选] `parser_type` : 
        - 解析类型，可手动指定也可根据`parser_param.length()`自动推断
        - 支持解析类型 : `var` / `bit` / `auto(默认缺省值)`
    - [可选] `description` : 注释及使用描述
- `data_array` : CAN协议数组解析规则
    - `can_package_num` : 预期接收的CAN数据帧数量
    - `can_id` : 预期接收作为数组值的CAN_ID数组
        - 手动指定所有CAN_ID(即`can_id.length() == can_package_num`) :
            - `can_id`为`array<string-HEX>[can_package_num]`，即长度为`can_package_num`的`十六进制字符串`
            - 所有数据将依照所指定的CAN_ID顺序，以数组`index`顺序装填，即通过CAN_ID查询在`can_id`中的`index`，按`array[index * 8]`为基准进行装填
            - 详细解析见下方示例表格(`例5`)
        - 自动指定连续CAN_ID(即`can_id.length() ！= can_package_num && can_id.length() == 2 && can_id[0] < can_id[1]`)
            - `can_id`为`array<string-HEX>[2]`，即长度为2的`十六进制字符串`
            - 所有数据将依照所指定的CAN_ID顺序，以数组`index`顺序装填，即通过CAN_ID查询在`can_id`中的`index`，按`array[index * 8]`为基准进行装填
            - 详细解析见下方示例表格(`例6`)
    - `array_name` : 需要解析到的数组名称(即在代码中使用`LINK_VAR(var)`链接的数组名称)
    - [可选] `description` : 注释及使用描述
- `cmd` : CAN协议下发指令解析规则(`例7`)
    - `cmd_name` : 指令名称(即在代码中使用`Operate()`函数进行操作的指令)
    - `can_id` : 指令传输所用CAN_ID
    - [可选] `ctrl_len` : 控制段长度，需满足`0 <= ctrl_len < MAX_CAN_LEN`，其中`MAX_CAN_LEN`在`STD_CAN`中为8，`FD_CAN`中为64，默认缺省值为 : `0`
    - [可选] `ctrl_data` : 
        - 控制段数据，为`十六进制字符串数组`(十六进制字符串不超过十六进制的两位(即u8))，满足`0 <= ctrl_data.length() <= ctrl_len`，默认缺省值为 : `[]`
    - [可选] `description` : 注释及使用描述

示例集:

> 例1:
>
> | 解析规则 | 数据 |
> | :----: | :----: |
> | var_name = "example_1" | can_data[0] = 0x12 |
> | var_type = "u16"       | can_data[1] = 0x34 |
> | parser_param = [1, 2]  | can_data[2] = 0x56 |
>
> 解析后等效代码为:
> ```cpp
> uint16 example_1 = 0x3456;
> ```
>
> 该示例展示了以参数`[1, 2]`，即`can_data`数组`index`为`1`和`2`共2个`u8`<u>以高位优先按二进制</u>的解析形式合并为`u16`类型变量`example_1`的过程

> 例2:
> 
> | 解析规则 | 数据 |
> | :----: | :----: |
> | var_name = "example_2" | can_data[0] = 0x12 |
> | var_type = "float"     | can_data[1] = 0x34 |
> | parser_param = [0, 3]  | can_data[2] = 0x56 |
> | var_zoom = 0.0001      | can_data[3] = 0x78 |
> 
> 解析后等效代码为:
> ```cpp
> uint32 tmp_2 = 0x12345678;
> float example_2 = *((float *)(&tmp_2)) * 0.0001;
> ```
>
> 该示例展示了以参数`[0, 3]`，即`can_data`数组`index`为`0`、`1`、`2`、`3`共4个`u8`<u>以高位优先按二进制</u>的解析形式合并为`float`类型变量`example_2`的过程

> 例3:
> 
> | 解析规则 | 数据 |
> | :----: | :----: |
> | var_name = "example_3" | can_data[0] = 0x12 |
> | var_type = "double"    | can_data[1] = 0x34 |
> | parser_param = [0, 3]  | can_data[2] = 0x56 |
> | var_zoom = 0.0001      | can_data[3] = 0x78 |
> 
> 解析后等效代码为:
> ```cpp
> uint32 tmp_2 = 0x12345678;
> double example_2 = (double)(tmp_2) * 0.0001;
> ```
>
> 该示例展示了以参数`[0, 3]`，即`can_data`数组`index`为`0`、`1`、`2`、`3`共4个`u8`先<u>以高位优先按二进制</u>的解析形式合并为`u32`，再以<u>小数点精度缩放</u>的解析形式，缩放小数点得到`double`类型变量`example_3`的过程
> ***
> __注__:在`var_type`为`float`或`double`中，当`parser_param`规定的解析`u8`数量不等于`flaot`和`double`的`size`且为`size`的偶数分之一(1/2, 1/4)，将使用<u>以高位优先按二进制</u>合并，再使用<u>小数点精度缩放</u>的解析形式:
> - 即 `float`的`size`为4个`u8`，当`parser_param`规定的解析`u8`数量为2时
> - 即 `double`的`size`为8个`u8`，当`parser_param`规定的解析`u8`数量为4或2时

> 例4:
> 
> | 解析规则 | 数据 |
> | :----: | :----: |
> | var_name = "example_4"   | can_data[0] = 0x12 (0b0001'0010) |
> | var_type = "u8"          | can_data[1] = 0x34 (0b00<u>11</u>'0100) |
> | parser_param = [1, 5, 4] | can_data[2] = 0x56 (0b0101'0110) |
> 
> 解析后等效代码为:
> ```cpp
> uint8 tmp_4 = 0x34;  // 0b0011'0100
> uint8 example_4 = (uint8)((0b0011'0100 & 0b0011'0000) >> 4);
> uint8 example_4 = 0b0000'0011;
> ```
>
> 该示例展示了以参数`[1, 5, 4]`，即`can_data`数组`index`为`1`，取高`5`位到低`4`位(&= 0b0011'0000)后右移`4`位

> 例5:
> 
> | 解析规则 | 0x200 | 0x201 | 0x2FF |
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
> 解析后等效代码为:
> ```cpp
> uint8 example_5[24] = {
>   0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
>   0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7,
>   0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7
> };
> ```
>
> 该示例展示了以参数`["0x201", "0x2FF", "0x200"]`，即按照`0x201`、`0x2FF`、`0x200`的顺序依次解析并装填入数组`example_5`

> 例6:
> 
> | 解析规则 | 0x200 | 0x201 | 0x202 |
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
> 解析后等效代码为:
> ```cpp
> uint8 example_6[24] = {
>   0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5, 0xA6, 0xA7,
>   0xB0, 0xB1, 0xB2, 0xB3, 0xB4, 0xB5, 0xB6, 0xB7,
>   0xC0, 0xC1, 0xC2, 0xC3, 0xC4, 0xC5, 0xC6, 0xC7
> };
> ```
>
> 该示例展示了以参数`["0x200", "0x202"]`，即按照从`0x200`到`0x202`的顺序依次解析并装填入数组`example_6`

> 例7:
>
> | 解析规则 | 数据 |
> | :----: | :----: |
> | cmd_name = "cmd_1"           | cmd_data[0] = 0x12 |
> | can_id = "0x1F2"             | cmd_data[1] = 0x34 |
> | ctrl_len = 4                 | cmd_data[2] = 0x56 |
> | ctrl_data = ["0xA1", "0xA2"] | cmd_data[3] = 0x78 |
>
> 使用方式:
> ```cpp
> can_protocol_1.Operate("cmd_1", cmd_data);
> ```
>
> 使用后等效代码为:
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
> 该示例展示了以`cmd_1`携带`cmd_data`下发指令，即按照`ctrl_len`规定前4个u8都为控制段数据并装填入`ctrl_data`，但因数据不足后2个u8留空，剩下4个u8按传入参数`cmd_data`填入
