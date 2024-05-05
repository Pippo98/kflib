cmake_minimum_required(VERSION 3.16)
project(KF)

set(DIR ${CMAKE_CURRENT_LIST_DIR})

add_subdirectory(eigen)
include_directories(
    ${DIR}
    ${DIR}/eigen
)

add_library(kflib
    SHARED
    ${DIR}/src/kf_impl.cpp
)
add_library(kflib_static
    STATIC
    ${DIR}/src/kf_impl.cpp
)

target_link_libraries(kflib PUBLIC Eigen3::Eigen)
target_link_libraries(kflib_static PUBLIC Eigen3::Eigen)
