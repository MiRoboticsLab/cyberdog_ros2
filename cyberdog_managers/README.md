# CyberDog 决策模块

---
## **[简介]**
---
该项目包含两个子项目：`decision_maker`和`decision_utils`。后者为决策相关功能的基类和工具类集合，负责实现通用功能；前者面向业务层，负责实现具体业务功能。

截至目前：

`decision_utils`中提供了`cascade_manager`，该模块继承于`cyberdog_utils::LifecycleNode`，具备级联/并联和单点控制的功能，可以快速管控其作用域下的节点的启动和关闭。

`decision_maker`中提供了`automation_manager`、`ception_manager`、`interaction_manager`和`motion_manager`。分别用于自动化功能、感知功能、人机交互功能和运动功能的管理和决策。该四个模块均继承`cascade_manager`，并在基础上根据业务功能稍作改动。

## **[功能特性]**
---
### **级联管理器**

#### **简介**

本功能实现于[decision_utils](decision_utils/src/cascade_manager.cpp)。目的是实现一个可以通过自启动和自暂停来并行管理其子节点状态的功能。它继承于`cyberdog_utils::LifecycleNode`，该节点的原理可以跳转至[cyberdog_utils](../cyberdog_common/cyberdog_utils)进行了解。

级联管理器按照`ROS 2`生命周期节点的机制，依次实现了`manager_configure`、`manager_activate`、`manager_deactivate`、`manager_cleanup`、`manager_shutdown`和`manager_error`，并需要继承它的节点在相应的过渡态调用相应的函数。支持参数配置。

#### **参数配置**

[] 内为变量名。

- 级联节点列表名[node_list_name_]，参数名根据输入字符串确定。
- 管理器超时时间[timeout_manager_]，参数名为`timeout_manager_s`。

#### **基本机理**

级联管理器通过辨识外部参数区分自身功能类型。目前共实现三种功能类型，分别为：`单管理器模式`、`单列表管理模式`和`多列表管理模式`。

- 单管理器模式：没有需要连接的子节点，所有的操作只考虑自身节点即可。
- 单列表管理模式：节点配置时，会通过级联节点列表名读取外部参数（YAML）列表，并根据列表去依次添加相应的节点名为子节点，并进行状态同步。且在激活和暂停的操作时，会检查列表中的每个节点是否达到相应的状态，并及时更新`chainnodes_state_`变量。
- 多列表管理模式：和单列表相似，节点配置时，会通过外部参数列表读取，只不过在配置时可以通过输入不同的参数列表进行读取，使用该管理模式需要在实例化时将级联节点列表名复制为`multi`以激活多列表管理模式，并在配置时向`manager_configure`函数传入列表名称的正确字符串。

所有继承级联管理器的节点均可使用`message_info`、`message_warn`和`message_error`进行ROS相关的记录输出，其中输入`std::string`格式的内容即可（多条字符串可用`+`进行组合）。

### **自动化管理**

#### **简介**

目前本节点只负责激活和暂停自动化相关节点，包括`Explor`模式和`Track`模式中特别需要的节点。支持参数配置。

#### **参数配置**

- 探索建图模式节点列表[explor_map_node_list_name_]，参数名根据输入字符串确定。
- 探索导航模式节点列表[explor_nav_node_list_name_]，参数名根据输入字符串确定。
- 跟踪模式节点列表[track_node_list_name_]，参数名根据输入字符串确定。

### **感知管理**

#### **简介**

目前本节点负责激活和暂停感知相关功能节点，并负责收集重要的感知信息，融合感知信息并对信息进行处理和分类。

#### **参数配置**

- 滴答频率[rate_tik_]，参数名为`rate_tik_hz`。
- 低电量阈值[soc_limit_]，参数名为`soc_limit_perc`。

#### 安全保护

目前安全保护功能以rate_tik_的频率检测机器人状态，目前仅支持低电量状态的辨别，并对外部节点进行广播通知。后续会增加新功能。

### **人机交互管理**

#### **简介**

目前本节点负责激活和暂停人机交互相关功能节点。

### **运动管理**

#### **简介**

目前本节点负责所有与运动相关的指令收集、指令处理和指令下发，所有的运动相关的状态收集、状态处理和状态广播，以及所有与运动相关的参数配置和功能配置。

- 指令包括：模式切换、步态切换、动作执行和行动指令。
- 状态包括：当前的模式、步态、缓存步态、动作、速度、位姿和参数等。
- 参数包括：步态参数和节点内参数。

由于目前运动控制层面是使用`Lightweight Communications and Marshalling（LCM）`进行通信，故本模块还会调用LCM相关接口进行通讯。

定制化的动作会通过`Tom's Obvious, Minimal Language（TOML）`进行设计，目前该模块是运动控制开发人员通过`Mini Cheetah`的框架进行改进后设计的，并提供`txt`格式，我们设计了一个`Python`脚本自动转换`txt`到`TOML`。*未来将开发自动生成动作`TOML`文件的功能，开发者并不需要手动编写这些文件。*

由于机器人只有一个主体，故下述的`模式切换`、`步态切换`和`动作执行`都具备抢占功能。

#### **节点参数配置**
##### 频率类
- 通用频率[rate_common_]
- 控制频率[rate_control_]
- 输出频率[rate_output_]
- 等待频率[rate_wait_loop_]
- 里程计频率[rate_odom_]
- LCM广播频率常量[rate_lcm_const_]
##### 超时类
- 管理器超时时间[timeout_manager_]
- 运动超时时间[timeout_motion_]
- 步态超时时间[timeout_gait_]
- 动作超时时间[timeout_order_]
- LCM接收超时时间[timeout_lcm_]
##### 赋值类
- X轴线速度极值[cons_abs_lin_x_]
- Y轴线速度极值[cons_abs_lin_y_]
- X轴角速度极值[cons_abs_ang_r_]
- Y轴角速度极值[cons_abs_ang_p_]
- Z轴角速度极值[cons_abs_ang_y_]
- Z轴角加速度极值[cons_abs_aang_y_]
- 身体高度最大值[cons_max_body_]
- 足底高度最大值[cons_max_gait_]
- 身体高度默认值[cons_default_body_]
- 足底高度默认值[cons_default_gait_]
- 默认移动线速度[cons_speed_l_normal_]
- 默认移动角速度[cons_speed_a_normal_]
##### 比例类
- 低电量控制比例[scale_low_btr_]
##### 通讯类
- 运动上返通道[ttl_recv_from_motion_]
- 运动发送通道[ttl_send_to_motion_]
- 里程计上返通道[ttl_from_odom_]
- 运动上返端口[port_recv_from_motion_]
- 运动发送端口[port_send_to_motion_]
- 里程计上返端口[port_from_odom_]

#### **运动参数配置**

数据格式：参考[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

运动参数修改目前仅支持身体高度和足底高度的动态修改。

合法请求：具备较新时间戳的运动参数数据。

- 较新时间戳表示此次参数修改的时间戳比上一次参数修改的时间戳要新，一般取同时区的当前系统时间即可。
- 参数范围在上述`cons_max_body_`和`cons_max_gait_`的约束内，且为正数。



#### **模式切换**

数据格式：参考[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

合法请求：具备较新时间戳的模式。
- 较新时间戳表示此次切换请求的时间戳比上一次模式切换的时间戳要新，一般取同时区的当前系统时间即可。
- 模式需要保证符合预设的选项，模式切换支持默认模式（MODE_DEFAULT）、锁定模式（MODE_LOCK）、手动模式（MODE_MANUAL）、探索模式（MODE_EXPLOR）和跟踪模式（MODE_TRACK）。模式分为`control_mode`和`mode_type`两个字段进行区分，前三个模式的`mode_type`均为`DEFAULT_TYPE（0）`，后两个模式可根据`mode_type`进行辨识不同的子模式。

抢占功能：新发起的合法模式切换请求均可抢占正在运行的旧的模式切换请求，即，新的优先级高。相同模式（两个字段都相同）不抢占。

取消功能：发起申请的句柄可以在任意时刻发起取消。

反馈：一旦收到合法请求，模式Action服务会以`rate_common_`Hz的频率检测当前切换状态，并以该频率返回当前切换情况。

结果：
- 切换成功，更新`robot_control_state_`中的`modestamped`值。
- 切换失败，则恢复上一状态。如切换过步态，则恢复上一个步态（被抢占除外）；如激活过其他节点，则暂停其他节点。

#### **步态切换**

数据格式：参考[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

合法请求：具备合法动机，且具备较新时间戳的步态
- 合法动机指的是，`motivation`取值必须符合预设值`cyberdog_utils::GaitChangePriority`
- 较新时间戳表示此次切换请求的时间戳比上一次步态切换的时间戳要新，一般取同时区的当前系统时间即可。
- 步态需要保证符合预设的选项，步态目前支持断电伏地（GAIT_PASSIVE)、缓慢趴下（GAIT_KNEEL）、恢复站立（GAIT_STAND_R)、姿态站立（GAIT_STAND_B）、缓慢行走（GAIT_WALK）、低速小跑（GAIT_SLOW_TROT）、中速小跑（GAIT_TROT)、快速小跑（GAIT_FLYTROT)、双腿蹦跳（GAIT_BOUND）和四腿蹦跳（GAIT_PRONK）。

抢占功能：首先考虑动机（优先级），其次考虑步态。
- 根据动机的值（uint8），从大到小，优先级从高到低。
- 同优先级如果步态不同，则新的请求抢占旧的请求。

取消功能：发起申请的句柄可以在任意时刻发起取消。

打断功能：如果在步态切换的过程中发生模式变更，则立即中止当前步态切换。

反馈：一旦收到合法请求，步态Action服务会以`rate_common_`Hz的频率检测当前切换的步态，并以该频率返回当前切换情况。

结果：
- 切换成功：更新`robot_control_state_`中的`gaitstamped`值。
- 切换失败：恢复`gait_cached_`的预设值。

#### **动作执行**

数据格式：参考[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

合法请求：具备较新时间戳的动作及其参数（如果需要）
- 较新时间戳表示此次执行请求的时间戳比上一次动作执行时间戳要新，一般取同时区的当前系统时间即可。
- 动作需要符合预设的选项，动作目前支持站起来（MONO_ORDER_STAND_UP）、趴下去（MONO_ORDER_PROSTRATE）、后退（MONO_ORDER_STEP_BACK）、原地转圈（MONO_ORDER_TURN_AROUND）、握手（MONO_ORDER_HI_FIVE）、跳舞（MONO_ORDER_DANCE）、拜年（MONO_ORDER_WELCOME）、打滚（MONO_ORDER_TURN_OVER）和坐下（MONO_ORDER_SIT）。
- 其中后退和原地转圈是可以设置参数的，即`para`字段给的值是有效的。动作执行效果会与参数直接相关。

抢占功能：除了拜年和打滚两个动作外，其他动作均可被新的动作执行请求抢占。

取消功能：除了拜年和打滚两个动作外，其他动作均可被发起申请的句柄在任意时刻取消。

打断功能：除了拜年和打滚两个动作外，其他动作都可以被步态切换请求、模式切换请求和移动指令直接打断。

外置调参：目前动作分为三类：
- ROS层直接封装：使用现有步态进行操作。
- ROS层设置参数表，运动控制层接收参数表后依次解析参数后执行：建立`TOML`的参数文件，读取文件后发送参数表。*目前建立和调优参数表的方式需要工程师做实验后手动调节，未来会设计一个自动化编程的软件。*
- ROS层只下发特定ID，运动控制层硬编码，直接执行：底层硬编码，两个不能打断和抢占的动作便是如此机理的。

反馈：一旦收到合法请求，动作执行服务会以`rate_common_`Hz的频率检测动作执行的进度（从0到100），并以该频率返回当前执行情况，同时更新`robot_control_state_`的`orderstamped`。

结果：
- 执行成功：更新`robot_control_state_`中的`id`为`MONO_ORDER_NULL`，恢复状态，返回成功。
- 执行失败：更新`robot_control_state_`中的`id`为`MONO_ORDER_NULL`，恢复状态，返回失败。

#### **行动指令**

数据格式：参考[motion_msgs](../cyberdog_interfaces/motion_msgs/README.md)

合法请求：具备恰当source id和指定frame id，且具备较新时间戳的行动指令
- 恰当source id：在不同的模式下，决策端接受的行动指令会通过source id的值进行分辨。如下所示，宏定义取自motion_msgs/msg/SE3VelocityCMD.msg。
  - 遥控模式：接受INTERNAL和REMOTEC
  - 探索模式：接受INTERNAL、REMOTEC和NAVIGATOR
  - 跟踪模式：接受INTERNAL和NAVIGATOR
- 指定的frame id：所有的行动指令的frame id必须为BODY_FRAME（取自motion_msgs/msg/Frameid.msg）
- 较新时间戳表示此次指令发起的时间戳比上一次行动指令的时间戳要新，一般取同时区的当前系统时间即可。

## **运行 & 调试**
---
本模块支持独立运行和启动系统运行，可结合GRPC或手柄两种控制模式。

### **独立运行**

独立运行即使用`ros2 run`启动，常用于单功能测试和单功能调试时使用。

完整指令为：

```
ros2 run cyberdog_decisionmaker decisionmaker
```

该启动状态下，由于没有外部参数，只能测试运动相关功能。

### **启动系统运行**

启动系统运行即使用`ros2 launch`启动，铁蛋会开机自动调用该脚本启动，除了启动决策节点外，还同时启动其他若干个节点，具体可参考[cyberdog_bringup](../cyberdog_bringup)进行深入了解。

完整指令为：

```
ros2 launch cyberdog_bringup lc_bringup_launch.py
```

该启动状态是机器人正常启动的流程，可以测试所有功能。

### **调试方法**

本模块支持内置调试模式，并支持GDB调试。

#### 内置调试模式

`motion_manager`会检测三个宏定义，分别是开启调试的总开关，行动调试和模拟运动数据。

```
DEBUG_ALL  // for complete debug
DEBUG_MOTION // for gait & motion debug
DEBUG_MOCK  // for mock lcm messages
```

进入调试模式，需要打开`DEBUG_ALL`：
- 如果需要输出移动相关的日志，可以打开`DEBUG_MOTION`
- 如果没有LCM数据输入源，可以打开`DEBUG_MOCK`
- `DEBUG_MOTION`和`DEBUG_MOCK`可以同时打开

可以通过在`decision_maker`的根目录建立`.debug_xxx`来开启调试功能，如

```shell
$ touch .debug_all
```

对应关系如下表

|DEF|File|
|---|----|
|DEBUG_ALL|.debug_all|
|DEBUG_MOTION|.debug_motion|
|DEBUG_MOCK|.debug_mock|

#### GDB调试

1. 首先需要在`CMakeLists.txt`中添加`-g`的编译标记，一般在`add_compile_options`函数里
2. 再根据[cyberdog_bringup](../cyberdog_bringup)进行修改，使用`gdb`前缀进行启动
3. 确定系统中是否具备调试终端工具，包括`gdb`和`xterm`等，如不具备需要安装
4. 确保系统中在相同`Domain ID`和相同`namespace`下不存在相同的节点后，在具备图形化界面的环境使用`Launch`进行启动。
5. 开始调试。

## **[未来优化]**
---
- 完善感知功能决策
- 完善交互功能决策
- 完善自动化功能决策
- 模块化 & 插件化单点功能
- 动态调参
- 动作的在线编程
- 功能的自定义配置