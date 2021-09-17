# Contributing to Xiaomi CyberDog！

Thanks for your interest in our project.

## About license

This project is under Apache 2.0.

## Before you contribute

Make sure you have signed Individual Contributor License Agreement (CLA). If you have signed, please read [Developer signature verification](#developer-signature-verification). If not, please continue to read.

Considering environmental protection and efficiency, currently only supported `Electronic Signatures`.

### Electronic signature Method

- Download [Individual  Contributor License Agreement](https://cdn.cnbj2m.fds.api.mi-img.com/cyberdog-package/packages/doc_materials/cla_zh_en.pdf)
- Use PDF to sign the agreement. Note that the signatures need to be handwritten (electronic signature).
- Send the signed agreement to the email[mi-cyberdog@xiaomi.com](mailto:mi-cyberdog@xiaomi.com)

## Developer Signature Verification

Developers require to use the email that signed the CLA to sign the submitted changes, which means using `git commit -s` to submit.

## Code Review

All submissions are processed according to Merge Request, and only accept processes similar to Pull Request on GitHub.

### Code Style

All codes related to ROS 2 follow `ROS 2` coding style. Please read [Code style and language versions](https://docs.ros.org/en/foxy/Contributing/Code-Style-Language-Versions.html) for more details. It is recommended that using [ament_lint](https://github.com/ament/ament_lint) tool or `colcon test` to check before submitting the code.

## Branch management

All our repositories are divided into two branches for development, which are `devel` and `main`.

- The `devel` branch is used for development.
- The `main` branch is used to store the stable version. If and only if a new version needs to be released, freeze the `devel` branch a from the `devel` branch, and perform the following steps:

  - Modify `CHANGELOG`, sort out new features and pre-fixed bugs in the pre-release, and determine  the release version.
  - Pass by CI, including construction and testing. 
  - The test engineer will fully test the new functions and issues proposed in the step 1. If there is a problem, the relevant developers are required to submit the repair code to the `devel` branch timely according to the method of fixing the problem.
  - If tests are passed, the code and binary package of the project will be packaged and encapsulated.
  - `Release` in the interface.
