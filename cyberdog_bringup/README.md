# cyberdog bringup

## 简述

该模块是启动所有其他子模块的启动模块。

## 总启动流程

该模块用于启动所有ROS 2进程，并直接应用于系统服务中。

## 总启动流程

完整的启动流程是通过[systemd service](https://git.n.xiaomi.com/mirp/cyberdog_ros2_deb/-/blob/master/src/etc/systemd/system/cyberdog_ros2.service)调用[lc_bringup_launch.py](https://git.n.xiaomi.com/mirp/cyberdog_bringup/-/blob/master/launch/lc_bringup_launch.py)启动其他所有的节点的启动脚本，并且`lc_bringup_launch.py`会调用`params`文件夹里的`launch_nodes.yaml`,`launch_groups.yaml`, `default_param.yaml`和`remappings.yaml`进行启动。在上述四个`YAML`文件中会配置`可以启动的节点`，`分组启动的节点`，`节点内需要使用的参数`和`节点内需要重映射`的内容。

在`launch_nodes.yaml`中会提供两种节点：`base_nodes`和`other_nodes`，前者是机器人运动和持续工作所必须的节点，如`不存在`或`启动失败`，则会上报错误，终止启动；后者是其他节点，`不存在`或`启动失败`只会上报警告，并继续启动。

在`launch_groups.yaml`中可按需配置不同条件下所需要的节点。

在`default_param.yaml`中会提供所有节点的外部参数，按照节点的名字进行分组。

在`remappings.yaml`中会提供所有节点所需要的重映射内容，按照节点的名字进行分组。

## 维护方式

基于上述描述，目前版本（210726）如需要添加或修改节点，不需要启动脚本`lc_bringup_launch.py`，只需要维护`launch_nodes.yaml`, `launch_groups.yaml`,`default_param.yaml`和`remappings.yaml`这四个`YAML`文件即可。下面会介绍如何使用这四个文件：

### launch_nodes.yaml文件使用

该文件用于配置需要启动的节点，主要结构如下：
```yaml
launch_nodes:
    debug_param: "lxterminal -e gdb -ex run --args"

    # Must launch, if package or executable error, launcher will stop and throw error
    base_nodes:

        # - example_node_def_name:
        #       package: "example_pkg"              (string)
        #       executable: "example_exe"           (string)
        #       #[optional] output_screen: false    (true/false)
        #       #[optional] name: 'example_name'    (string)
        #       #[optional] load_nodes_param: false (true/false)
        #       #[optional] enable_debug: false     (true/false)

    # Try to launch, if package or executable error, launcher will notice but skip that node
    other_nodes:
```
- `launch_nodes`内包含1个测试参数`debug_param`, 2个启动节点配置类型(`base_nodes`, `other_nodes`)

测试参数：
- 在节点中配置`enable_debug == true`时，`debug_param`参数将写入ros2 Node中prefix

配置类型：
- 配置类型分为`base_nodes`与`other_nodes`
- `base_nodes` 为基础节点，是核心功能节点，当这些节点启动失败时将会报错并`停止`所有其他节点启动
- `other_nodes`为辅助节点，是其他功能节点，当这些节点启动失败时将会提醒并`跳过`该出错节点继续启动

在`base_nodes`与`other_nodes`中即可配置需要启动的节点

定义节点配置名称(任意名称)：
```
# - example_node_def_name:
```
配置节点启动的包名(根据软件包)：
```
#       package: "example_pkg"              (string)
```
配置节点启动的可执行文件名(根据软件包)：
```
#       executable: "example_exe"           (string)
```
配置节点标准输出位置(可缺省)：
```
#       #[optional] output_screen: false    (true/false)
```
配置节点名称(可缺省)：
```
#       #[optional] name: 'example_name'    (string)
```
配置是否装载外部参数(从`params/default_param.yaml`载入，可缺省)：
```
#       #[optional] load_nodes_param: false (true/false)
```
配置是否装载debug参数(可缺省)
```
#       #[optional] enable_debug: false     (true/false)
```

### launch_groups.yaml文件使用

该文件用于配置启动组，主要结构如下：
```yaml
launch_groups:
  target_launch_group: default
  groups:
    # example_group_whitlist:
    #   launch:   #only launch:
    #     - example_node_def_name1
    #     - example_node_def_name2
    # example_group_blacklist:
    #   except:   #dont launch:
    #     - example_node_def_name1
    #     - example_node_def_name2
    default:
      except:
        - cyberdog_motion_test_cmd
        - decision_test
```
- `target_launch_group`为当前启动组
- `groups`中配置启动组

在`groups`中有两种配置方式，分别为`launch`和`except`，但一个启动组中只能有一种启动方式
- `launch`:只启动所列出的全部节点
- `except`:启动除了列出的全部节点

### default_param.yaml文件使用

该文件用于参数设置，主要结构如下：

```yaml
# 节点名
example_node_def_name:
  # 固定的，表明为ROS参数
  ros__parameters:
    # 参数名 和 值
    value_a: 1
```

具体细节可参考[ROS2 YAML For Parameters](https://roboticsbackend.com/ros2-yaml-params/)

### remapping.yaml文件使用
该文件用于重映射，主要结构如下：
```yaml
remappings:
  example_node_def_name:
    - ['/topic_before1','/topic_after1']
    - ['/topic_before2','/topic_after2']
```
其中`example_node_def_name`根据`launch_nodes.yaml`文件中定义的节点名填写




