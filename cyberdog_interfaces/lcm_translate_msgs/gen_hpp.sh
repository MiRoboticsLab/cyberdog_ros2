#!/bin/bash

if [ -d $1 ]; then
  rm -rf $1
  echo "$1 exist, cleanup it"
fi

lcm-gen -x ${@:2} --cpp-hpath $1