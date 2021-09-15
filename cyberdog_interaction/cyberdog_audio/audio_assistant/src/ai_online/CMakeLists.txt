cmake_minimum_required(VERSION 3.4)
project(AivsSample)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    add_definitions("-std=c++11")
else()
    add_definitions("-std=c++0x -Dnullptr=NULL -DNOT_SUPPORTS_CXX11")
endif()

include(CheckIncludeFile)
check_include_file(sys/capability.h HAVE_SYS_CAPABILITY)
if(HAVE_SYS_CAPABILITY)
set(CAP cap)
endif()

include(CheckLibraryExists)
check_library_exists(z deflate "" HAVE_ZLIB)
if(HAVE_ZLIB)
set(ZLIB z)
endif()

check_include_file(pthread.h HAVE_PTHREAD)
if(HAVE_PTHREAD)
set(PTHREAD pthread)
endif()

include_directories(
    .
    ../include
)

link_directories(
    ../lib
)

add_executable(
    AivsSample

    AivsSample.cpp
    AuthCapabilityImpl.cpp
    StorageCapabilityImpl.cpp
    ConnectionCapabilityImpl.cpp
    ErrorCapabilityImpl.cpp
)

target_link_libraries(
    AivsSample

    aivs_sdk
    ${CAP}
    ${ZLIB}
    ${PTHREAD}
)