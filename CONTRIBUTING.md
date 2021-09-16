# 向小米铁蛋贡献力量！ - Contributing to Xiaomi CyberDog！

感谢您关注我们的项目。

Thanks for your interest in our project.

## 关于协议 - About license

本项目遵守Apache 2.0开源协议。

This project is under Apache 2.0.

## 成为开发者前 - Before you contribute

确保您已经签署个人独立贡献者许可协议（CLA）。如果已签署，请直接阅读[开发者签名认证](#开发者签名认证)，如未签署，请继续往下读。

Make sure you have signed Individual Contributor License Agreement (CLA). If you have signed, please read [Developer signature verification](#developer-signature-verification). If not, please continue to read.

为环保和效率起见，目前只支持`电子签署`。

Considering environmental protection and efficiency, currently only supported `Electronic Signatures`.

### 电子签署方式 - Electronic signature Method

- 下载[个人独立贡献者许可协议](https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/doc_materials/cla_zh_en.pdf)
- Download [Individual  Contributor License Agreement](https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/doc_materials/cla_zh_en.pdf)
- 使用PDF软件签署协议，注意签名部分需要手写（电子签名即可）。
- Use PDF to sign the agreement. Note that the signatures need to be handwritten (electronic signature).
- 将签署完毕的协议文件发送至邮箱[mi-cyberdog@xiaomi.com](mailto:mi-cyberdog@xiaomi.com)
- Send the signed agreement to the email[mi-cyberdog@xiaomi.com](mailto:mi-cyberdog@xiaomi.com)

## 开发者签名认证 - Developer Signature Verification

开发者需要使用签署CLA的邮箱对所提交修改进行签名，即使用`git commit -s`进行递交commit信息。

Developers require to use the email that signed the CLA to sign the submitted changes, which means using `git commit -s` to submit.

## 代码审查 - Code Review

所有递交都按照合入请求（Merge Requst）的方式进行，并只接收GitHub的拉取请求（Pull Request）的流程方式。

All submissions are processed according to Merge Request, and only accept processes similar to Pull Request on GitHub.

### 代码格式 - Code Style

所有ROS 2相关代码均遵从`ROS 2`的标准代码规范。可阅读[Code style and language versions](https://docs.ros.org/en/foxy/Contributing/Code-Style-Language-Versions.html)进行了解。建议在提交代码前，使用[ament_lint](https://github.com/ament/ament_lint) 工具或借助`colcon test`进行快速审查。

All codes related to ROS 2 follow `ROS 2` coding style. Please read [Code style and language versions](https://docs.ros.org/en/foxy/Contributing/Code-Style-Language-Versions.html) for more details. It is recommended that using [ament_lint](https://github.com/ament/ament_lint) tool or `colcon test` to check before submitting the code.

## 分支管理 - Branch management

我们所有的仓库均分为两条分支进行开发，分别是`devel`和`main`。

All our repositories are divided into two branches for development, which are `devel` and `main`.

- `devel`分支用于日常的合入请求和每日版本的打包，属于开发分支。
- The `devel` branch is a develop
requests and bundle repositories.
- `main`分支用于存放稳定版本，当且仅当需要发行新版本时，冻结`devel`分支，并从`devel`分支创建合入请求，并经过：.
- The `main` branch is used to store the stable version. 
version needs to be released, freeze the `devel` branch a
from the `devel` branch, and perform the following steps:

  - 撰写改动记录，梳理预发行的新功能和预修复的问题，确定版本号。
  - Modify `CHANGELOG`, sort out new features and pre-fixed bugs in the pre-release, and determine  the release version.
  - CI完全通过，包括构建和测试部分。
  - Pass by CI, including construction and testing. 
  - 测试工程师介入，并按照1中的新功能和预修复的问题进行摸底测试。如有问题，需要相关开发者按照修复问题的方式及时向`devel`分支递交修复代码。
  - The test engineer will fully test the new functions and issues proposed in the step 1. If there is a problem, the relevant developers are required to submit the repair code to the `devel` branch timely according to the method of fixing the problem.
  - 如测试通过，则对项目的代码和二进制包进行打包和封装。
  - If tests are passed, the code and binary package of the project will be packaged and encapsulated.
  - 在`Release`界面进行发布。
  - `Release` in the interface.
