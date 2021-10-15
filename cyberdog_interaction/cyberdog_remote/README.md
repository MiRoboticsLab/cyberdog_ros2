# cyberdog_remote

该模块主要提供多种类型手柄的控制支持(xbox, ps5测试通过)，手柄的按键重映射，控制指令的键位设置。

[注意] Nintendo Switch Pro Controller 测试未通过， 操作系统无法正常驱动手柄，jstest获取设备数据异常，暂不支持直接使用，如有使用需求请自行配置驱动。

## 使用手册

- 遥控方法：
  - 通过数据线将手柄连接cyberdog外设扩展口(Extension)
  - [选做] 进行手柄按键重映射(仅首次连接无按键映射文件手柄需要(xbox, play station已内置按键映射文件)，否则将默认使用xbox按键映射文件，不确定键位是否正常)
    - 首次或在不熟悉的情况下可使用`ssh`命令进入cyberdog终端，使用`journalctl -f | grep cyberdog_remote`命令实时查看状态
    - 同时长按右侧4个主按键(xbox为`X`,`Y`,`A`,`B`; play station为`O`,`X`,`口`,`三角`)5s以上进入重映射模式(如手柄及驱动支持，成功进入会短暂震动提示)
    - 重映射按键，分别依次按下：(如手柄及驱动支持，每步按下均会短暂震动提示)
      - 右侧主按键区(上、下、左、右)
      - 左侧十字按键区(上、下、左、右)
      - 右侧扳机区(上、下)
      - 左侧扳机区(上、下)
      - 菜单按钮(右，左)
      - 摇杆(右，左)
    - 接着重映射摇杆，分别依次推动摇杆：(如手柄及驱动支持，每步操作均会短暂震动提示)
      - 右侧摇杆(上下，左右)
      - 左侧摇杆(上下，左右)
    - 完成摇杆重映射，配置文件将以`手柄名称.toml`的形式保存到`params/joys`目录下
  - 根据配置决定是否需要解锁：(手柄仅首次连接后需解锁(`need_unlock`配置为`true`, `need_unlock_everytime`配置为`false`)/手柄每次连接后都需要重新解锁(`need_unlock`与`need_unlock_everytime`同时配置为`true`))
    - 连续步骤：
      1. 将手柄左右摇杆一起，以前方为起点与终点，右摇杆顺时针，左摇杆逆时针，两摇杆同时反相位转动一圈，后返回中间位置即完成解锁步骤
      2. 完成解锁(如手柄及驱动支持，完成解锁会有强弱强的震动提示)
    - 分解步骤：
      1. 将手柄摇杆同时推至最前端(如手柄及驱动支持，每步解锁均会短暂震动提示)
      2. 将左侧摇杆推至最左端、同时右侧摇杆推至最右端
      3. 将手柄摇杆同时推至最后端
      4. 将左侧摇杆推至最右端、同时右侧摇杆推至最左端
      5. 将手柄摇杆同时推至最前端
      6. 松手或使摇杆都返回中间状态
      7. 完成解锁(如手柄及驱动支持，完成解锁会有强弱强的震动提示)
  - 键位操作：
    - 通过`上十字键`切换为手动模式`motion_msgs::msg::Mode::MODE_MANUAL`，通过`下十字键`切换趴下模式`motion_msgs::msg::Mode::MODE_DEFAULT`
    - `右菜单键`为断电急停`motion_msgs::msg::Mode::MODE_LOCK`
    - 通过`左右十字键`切换步态，默认范围为`motion_msgs::msg::Gait::GAIT_STAND_R`到`motion_msgs::msg::Gait::GAIT_PRONK`
    - 根据配置文件决定移动控制是否需要按住`使能键`(`enable_control`默认为`左侧下扳机键`)
    - 右侧`主按键区`(`action0`～`action3`默认为`Y`,`A`,`X`,`B`)为动作按键，按下后执行对应动作(`params/cyberdog_conf.toml`和`params/mono_order.toml`共同决定动作)
    - 根据配置的模式不同，`左右翻页键`(`last_page`/`next_page`默认为`左右侧上扳机键`)功能不同：
      1. 在`enable_muilt_page`为`false`时：仅提供3x4一共12个动作指令，分别为：按住`左翻页键`(`last_page`默认为`左侧上扳机键`)+主按键区、主按键区、按住`右翻页键`(`next_page`默认为`右侧上扳机键`)+主按键区
      2. 在`enable_muilt_page`为`true`时：提供无上限动作指令，通过`左右翻页键`(`last_page`/`next_page`默认为`左右侧上扳机键`)进行翻页后(页码不循环)，再按下对应`主按键区`(`action0`～`action3`默认为`Y`,`A`,`X`,`B`)按键执行，在该模式下可通过`页数重置键`(`reset_page`默认为`右侧下扳机键`)返回首页
    - 同时按下`左右摇杆`可进入手柄输入锁定模式
    - 摇杆操作与App控制相同

- 完整流程：
  - 连接手柄到cyberdog
  - (重映射手柄键位)
  - 解锁手柄
  - 切换手动模式
  - 切换步态
  - (按下控制使能键)摇杆操控移动 或 执行动作

## 配置手册

该模块拥有3种配置文件：主配置文件`cyberdog_conf.toml`、动作指令配置文件`mono_order.toml`、手柄按键映射文件`params/joys`目录下

- 主配置文件`cyberdog_conf.toml`：

  主配置文件分为基础配置、控制参数、功能按键映射、动作指令映射

  - 基础配置：
    - `min_callback_ms`：手柄最小反馈间隔(保证CPU负载不会过大)
    - `need_unlock`：是否需要操作摇杆解锁
    - `need_unlock_everytime`：是否在每次手柄插入后都需要操作摇杆解锁
    - `require_enable_button`：是否在操作cyberdog移动时需按下使能
    - `show_joyconfig`：是否在日志中输出当前连接手柄配置情况
    - `show_raw_joydata`：是否在日志中输出原始手柄数据`sensor_msgs::msg::Joy`
    - `show_self_joydata`：是否在日志中输出映射后的手柄数据
    - `unlock_max`：解锁时最高点判定大小(越小越容易，误触发越容易)
    - `unlock_min`：解锁时范围判定大小(越大越容易，误触发越容易)
    - `enable_muilt_page`：是否使用多页动作指令
    - `load_para`：载入的`控制参数名`
    
  - 控制参数：

    [控制参数名]
    - `scale_linear`：[x:前后, y:左右, z:无]，线速度缩放值(基于满摇杆1.0相乘)
    - `scale_angular`：[pitch, yaw, roll]，角速度缩放值(基于满摇杆1.0相乘)
    
  - 功能按键映射：

    主按键区：  `Main_U`   `Main_D`   `Main_L`   `Main_R`
    十字按键区：`Cross_U`  `Cross_D`  `Cross_L`  `Cross_R`
    扳机按键区：`Trig_R`   `Trig_RS`  `Trig_L`   `Trig_LS`
    菜单按键区：`Menu_R`   `Menu_L`
    摇杆按键区：`Stick_R`  `Stick_L`

    [func_key]

    - `enable_control`：使能键(默认为`左侧下扳机键`, `Trig_LS`)
    - `last_page`：左翻页键(默认为`左侧上扳机键`, `Trig_L`)
    - `next_page`：右翻页键(默认为`右侧上扳机键`, `Trig_R`)
    - `reset_page`：页数重置键(默认为`右侧下扳机键`, `Trig_RS`)
    - `action0`：动作键0(默认为`上主按键`, `Main_U`)
    - `action1`：动作键1(默认为`下主按键`, `Main_D`)
    - `action2`：动作键2(默认为`左主按键`, `Main_L`)
    - `action3`：动作键3(默认为`右主按键`, `Main_R`)
    
  - 动作指令映射:

    `enable_muilt_page`为`false`时将仅支持3个`action_order`，多余无效

    [[action_order]]

    - action0：动作0所对应的`MONO_ORDER`(定义在`mono_order.toml`)
    - action0_para：动作0所对应的参数(部分order可省略)
    - action1：动作1所对应的`MONO_ORDER`(定义在`mono_order.toml`)
    - action1_para：动作1所对应的参数(部分order可省略)
    - action2：动作2所对应的`MONO_ORDER`(定义在`mono_order.toml`)
    - action2_para：动作2所对应的参数(部分order可省略)
    - action3：动作3所对应的`MONO_ORDER`(定义在`mono_order.toml`)
    - action3_para：动作3所对应的参数(部分order可省略)

- 动作指令配置文件`mono_order.toml`：

  该文件用于保存动作指令的ID

  [order_list]

  MONO_ORDER = ID

- 手柄按键映射文件`params/joys`目录下：

  该目录用于存放自动remapping的按键映射文件，常规情况不会手动修改
