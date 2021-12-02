





### CAN通信描述文件使用

示例：

```toml
# -- common params -- #
# protocol = "can"          (string)
# name = "can_device_name"  (string)
protocol = "can"
name = "can_device_1"

# -- can params -- #
# can_interface = "can0"             (string)
# [optional] extended_frame = false  (true / false)
# [optional] canfd_enable = false    (true / false)
# [optional] timeout_us = -1         (int64)
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

解析：
- `common params` : 通用参数
    - `protocol` : 该描述文件所规定的通信协议
    - `name` : 该设备`common_device`的名称，在log中会使用该值来提示
- `can params` : CAN协议参数
    - `can_interface` : 使用的CAN总线，一般为`"can0"`, `"can1"` 或 `"can2"`
    - [可选] `extended_frame` : 是否使用扩展帧(主要针对发送，接收是全兼容的)，默认缺省值为 : `false`
    - [可选] `canfd_enable` : 是否使用CAN_FD(需要系统设置及CAN收发器硬件支持)，默认缺省值为 : `false`
    - [可选] `timeout_us` : 接收或发送超时时间(微秒)，主要用于防止析构时卡在接收或发送函数中无法进行，默认缺省值 : `-1(无超时)`
- `data_var` : CAN协议变量解析规则
    - `can_id` : 需要接收的ID，以`"0x"`或`"0X"`开头的`十六进制字符串`，可使用符号`“’”(单引号)`和`“ ”(空格)`进行任意分割以方便阅读和书写，如 : `0x1FF'12'34` 或 `0x123 45 67` 
    - `var_name` ： 需要解析到的变量名称(即在代码中使用`LINK_VAR(var)`链接的变量名称)
    - `var_type` ：
        - 需要解析的变量类型(即在代码中使用`LINK_VAR(var)`链接的变量类型)，在及其特殊情况下可使用不同类型(如将`u8`解析到代码中的`u16`)，但`size_of`的大小不能超过代码中的大小(即不可将`u64`解析到代码中的`u8`)
        - 支持解析格式 : `float` / `double` / `i64` / `i32` / `i16` / `i8` / `u64` / `u32` / `u16` / `u8` / `bool`
    - `parser_param` ： 解析参数，根据不同解析类型会存在不同长度
        - 以变量形式解析(即`parser_type = "var"`) :
            - `parser_param`为`array<uint8>[2]`，即长度为2的`uint8`数组，且满足`0 <= parser_param[0] <= parser_param[1] < MAX_CAN_LEN`，其中`MAX_CAN_LEN`在`STD_CAN`中为8，`FD_CAN`中为64
            - 解析时优先<u>以高位优先按二进制</u>解析原则将`can_data[parser_param[0]]`到`can_data[parser_param[1]]`以二进制形式合并为一个变量
            - `float`与`double`在特殊情况下先<u>以高位优先按二进制</u>的解析形式合并，再以<u>小数点精度缩放</u>的解析形式生成为一个变量
            - 详细解析见下方示例表格(`例1`到`例3`)
        - 以位形式解析(即`parser_type = "bit"`) :
            - `parser_param`为`array<uint8>[3]`，即长度为3的`uint8`数组，且满足`0 <= parser_param[0] < MAX_CAN_LEN` && `8 > parser_param[1] >= parser_param[2] >= 0`，其中`MAX_CAN_LEN`在`STD_CAN`中为8，`FD_CAN`中为64
            - 解析时通过<u>移位和位与</u>解析原则将`can_data[parser_param[0]]`中高`parser_param[1]`位到低`parser_param[2]`位中的数据取出并右移`parser_param[2]`位
            - 详细解析见下方示例表格(`例4`)
    - [可选] `var_zoom` : 缩放参数，以乘法形式与变量合并，仅`float`和`double`有效，默认缺省值为 : `1.0`
    - [可选] `parser_type` : 
        - 解析类型，可手动指定也可根据`parser_param.length()`自动推断
        - 支持解析类型 : `var` / `bit` / `auto(默认缺省值)`
    - [可选] `description` : 注释及使用描述
- `data_array` : CAN协议数组解析规则
    - `can_package_num` ： 预期接收的CAN数据帧数量
    - `can_id` ： 预期接收作为数组值的CAN_ID数组
        - 手动指定所有CAN_ID(即`can_id.length() == can_package_num`) :
            - `can_id`为`array<string-HEX>[can_package_num]`，即长度为`can_package_num`的`十六进制字符串`
            - 所有数据将依照所指定的CAN_ID顺序，以数组`index`顺序装填，即通过CAN_ID查询在`can_id`中的`index`，按`array[index * 8]`为基准进行装填
            - 详细解析见下方示例表格(`例5`)
        - 自动指定连续CAN_ID(即`can_id.length() ！= can_package_num && can_id.length() == 2 && can_id[0] < can_id[1]`)
            - `can_id`为`array<string-HEX>[2]`，即长度为2的`十六进制字符串`
            - 所有数据将依照所指定的CAN_ID顺序，以数组`index`顺序装填，即通过CAN_ID查询在`can_id`中的`index`，按`array[index * 8]`为基准进行装填
            - 详细解析见下方示例表格(`例6`)
    - `array_name` ： 需要解析到的数组名称(即在代码中使用`LINK_VAR(var)`链接的数组名称)
    - [可选] `description` : 注释及使用描述
- `cmd` :

> 例1：
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

> 例2：
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

> 例3：
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
> __注__：在`var_type`为`float`或`double`中，当`parser_param`规定的解析`u8`数量不等于`flaot`和`double`的`size`且为`size`的偶数分之一(1/2, 1/4)，将使用<u>以高位优先按二进制</u>合并，再使用<u>小数点精度缩放</u>的解析形式：
> - 即 `float`的`size`为4个`u8`，当`parser_param`规定的解析`u8`数量为2时
> - 即 `double`的`size`为8个`u8`，当`parser_param`规定的解析`u8`数量为4或2时

> 例4：
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
