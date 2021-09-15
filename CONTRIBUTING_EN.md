# Contributing to Xiaomi CyberDog

Thanks for your interest in our project.

## About license

This project is under Apache 2.0.

## Before you contribute

Make sure you have signed Individual Contributor License Agreement (CLA). If you have signed, please read [Developer signature verification](https://partner-gitlab.mioffice.cn/cyberdog/cyberdog_repos/-/blob/master/CONTRIBUTING.md#%E5%BC%80%E5%8F%91%E8%80%85%E7%AD%BE%E5%90%8D%E8%AE%A4%E8%AF%81). If not, please continue to read.

Considering environmental protection and efficiency, currently only supported "electronic signatures".

## Electronic signature Method

1. Download [Individual  Contributor License Agreement] (https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/doc_materials/cla_zh_en.pdf)
2. Use PDF to sign the agreement. Note that the signatures need to be handwritten (electronic signature).
3. Send the signed agreement to the email[mi-cyberdog@xiaomi.com](mailto:mi-cyberdog@xiaomi.com)

## Developer Signature Verification

Developers require to use the email that signed the CLA to sign the submitted changes, which means using `git commit -s` to submit.

## Code Review

All submissions are processed according to Merge Request, and only accept processes similar to Pull Request on GitHub.

### Code Style

All codes related to ROS 2 follow ROS 2 coding style. Please read [Code style and language versions](https://docs.ros.org/en/foxy/Contributing/Code-Style-Language-Versions.html) for more details. It is recommended that using `ament_lint`(https://github.com/ament/ament_lint) tool or `colcon test` to check before submitting the code.

## Branch management

All our repositories are divided into two branches for development, which are `devel` and `master`.

- The `devel` branch is a development branch and it is used for daily merge requests and bundle repositories.
- The `master` branch is used to store the stable version. If and only when a new version needs to be released, freeze the `devel` branch and create a merge request from the `devel` branch, and perform the following steps:
 1. Modify `CHANGELOG`, sort out new features and pre-fixed bugs in the pre-release, and determine the release version.
 2. Pass by CI, including construction and testing. 
 3. The test engineer will fully test the new functions and issues proposed in the step 1. If there is a problem, the relevant developers are required to submit the repair code to the `devel` branch timely according to the method of fixing the problem.
 4. If step 3 is passed, the code and binary package of the project will be packaged and encapsulated.
 5. Release in the interface.
