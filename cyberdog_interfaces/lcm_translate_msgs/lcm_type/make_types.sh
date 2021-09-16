#!/bin/bash


GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo -e "${GREEN} Starting LCM type generation...${NC}"

cd ../cyberdog_lcm/lcm_translate_msgs/lcm_type

# Clean
rm -rf dreame_dog
rm */*.jar
rm */*.java
rm */*.hpp
rm */*.class
rm */*.py
rm */*.pyc

no_java=false
while getopts 'c' OPT; do
  case $OPT in
    c)
      no_java=true
  esac
done

# Make
lcm-gen -jxp *.lcm

if [ "$no_java" = false ] ; then
  echo "with JAVA"
  cp /usr/local/share/java/lcm.jar .
  javac -cp lcm.jar */*.java
  jar cf my_types.jar */*.class
  mkdir -p java
  mv my_types.jar java
  mv lcm.jar java
fi

FILES=$(ls */*.class)
echo ${FILES} > file_list.txt


echo -e "${GREEN} Done with LCM type generation${NC}"
