# motion_msgs

该仓包含所有运动相关的ROS消息定义，包括所有直接触发行动<sup>1</sup>，直接影响行动<sup>2</sup>和行动上返<sup>3</sup>的消息。消息包括`action`和`msg`两种定义。

为了更好地定制化产品，我们在消息定义中尽可能地重新定义了所有使用到的消息，并避免使用动态数据（如不定长数组和字符串等类型）。优点是提高了产品接口的一致性和功能定制的便利性，缺点是损失了兼容性。

## Actions (.action)

- [ChangeGait.action](action/ChangeGait.action): 用于机器人切换步态，由于切换过程并非立即回复结果，故设计为`action`，在切换过程中会不断回复当前切换的步态，直至切换完成，回复结果和错误代码，以及发生错误的步态。

  - Request:
    - `motivation`[uint8] : 切换步态的动机，用于优先级判断。服务端只接受在`cyberdog_utils::GaitChangePriority`枚举过的动机。步态切换按照该值从大到小排列优先级，优先级高者优先执行。
    - `gaitstamped`[Gait] : 带有时间戳的步态消息。只有带有新的时间戳的步态请求才能被服务端接受。
  - Feedback:
    - `current_checking`[Gait] : 当前切换的步态。
  - Result:
    - `err_code`[uint8] : 步态切换的错误代码，错误彼此互斥，当有错误发生时，立即结束步态切换。错误定义在`action`中使用宏定义配置，宏定义名已语义化，不赘述。
    - `err_gait`[Gait] : 错误发生时所切换的步态。
    - `succeed`[bool] : 切换结果，`true`代表成功，`false`代表失败。

- [ChangeMode.action](action/ChangeMode.action): 用于机器人切换模式，同步态，在切换模式时也不会立即回复结果，且在切换不同模式时，其内部操作也有差异，故其回复的反馈内容也会随之改变。

  - Request:
    - `modestamped`[Mode] : 带有时间戳的模式消息。只有带有新的时间戳的模式请求才能被服务端接受。
  - Feedback:
    - `timestamp`[Time] : 发送反馈内容时刻的时间戳。
    - `current_state`[uint8] : 切换过程中的状态，目前包括`切换步态`、`等待节点启动`和`切换子模式`，宏定义在`action`内。
  - Result:
    - `err_code`[uint8] : 模式切换的错误代码，错误彼此互斥，当有错误发生时，立即结束模式切换。错误定义在`action`中使用宏定义配置，宏定义名已语义化，不赘述。
    - `err_state`[uint8] : 错误发生时，服务端的状态，与`current_state`宏定义相同。
    - `succeed`[bool] : 切换结果，`true`代表成功，`false`代表失败。

- [ExtMonOrder.action](action/ExtMonOrder.action): 用于机器人的额外动作执行，同上，每个动作的执行耗时不同，故需要在执行过程中不断返回当前机器人的状态。

  - Request:
    - `orderstamped`[MonOrder] : 带有时间戳的动作消息。只有带有新的时间戳的动作执行请求才能被服务端接受。
  - Feedback:
    - `order_executing`[MonOrder] : 当前执行的动作。
    - `process_rate`[float64] : 动作执行的进度条，取值从0.0到100.0。
    - `current_pose`[SE3Pose] : 执行当前时刻机器人本体躯干的位姿。
  - Result:
    - `err_code`[uint8] : 动作执行的错误代码，错误彼此互斥，当有错误发生时，立即结束动作执行。错误定义在`action`中使用宏定义配置，宏定义名已语义化，不赘述。
    - `succeed`[bool] : 执行结果，`true`代表成功，`false`代表失败。

## Messages (.msg)

- [ActionRequest.msg](msg/ActionRequest.msg): 暂时用于GRPC转发中的数据转换。
- [ActionRespond.msg](msg/ActionRespond.msg): 暂时用于GRPC转发中的数据转换。
- [ControlState.msg](msg/ControlState.msg): 机器人控制状态，包括当前时刻，当前模式（Mode）、步态（Gait）、缓存步态（Gait）、执行动作（MonOrder）、参数数据（Parameters）、速度（SE3Velocity）、位姿（SE3Pose）、安全级别（Safety）、场景（Scene，暂未支持）、错误状态（ErrorFlag）和接触地面状态（int8）。
- [Frameid.msg](msg/Frameid.msg): 整形定义的坐标系ID。包括用于向机器人发送行动指令的`BODY_FRAME`，用于感知、定位和导航的`ODOM_FRAME`, `VISION_FRAME` and `NAVI_FRAME`。当前程序只接受该文件定义中的`frame id`。
- [Gait.msg](msg/Gait.msg): 步态定义，参数包括时间戳（Time）和步态值（uint8）。内置默认支持的若干步态的宏定义。
- [MonOrder.msg](msg/MonOrder.msg): 单动作定义，参数包括时间戳（Time）和步态值（uint8）。内置默认支持的若干预设动作的宏定义。
- [Parameters.msg](msg/Parameters.msg): 机器人的实时参数，目前支持了步态的`躯干高度`和`足底高度`。
- [Safety.msg](msg/Safety.msg): 机器人的安全等级，目前仅支持`电量不足`。
- [Scene.msg](msg/Scene.msg): 场景分类，目前仅支持通过全球定位系统辨别`室内`和`室外`。
- [SE3Pose.msg](msg/SE3Pose.msg) : 特殊欧式群SE(3)定义下的位姿数据，包含时间戳、笛卡尔坐标系的三轴位置和三轴姿态。
- [SE3Velocity.msg](msg/SE3Velocity.msg): 特殊欧式群SE(3)定义下的速度数据，包含时间戳、笛卡尔坐标系的三轴线速度和三轴角速度。
- [SE3VelocityCMD.msg](msg/SE3VelocityCMD.msg): 行动命令消息，在`SE3Velocity`的基础上添加了`sourceid`字段用于区分指令来源。目前仅支持发送速度数据，未来会支持带有各类参数的行动命令。