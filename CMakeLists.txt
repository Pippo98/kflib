cmake_minimum_required(VERSION 3.16)
project(KF)

set(DIR ${CMAKE_CURRENT_LIST_DIR})

if(NOT TARGET Eigen3::Eigen)
  add_subdirectory(${DIR}/eigen)
  include_directories(
      ${DIR}/eigen
  )
  message(STATUS "Eigen3::Eigen added from ${DIR}/eigen")
endif()

add_library(kflib_object
  OBJECT
  ${DIR}/src/kf_impl.cpp
)
set_target_properties(kflib_object PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(kflib_object PUBLIC Eigen3::Eigen)

add_library(kflib_static STATIC $<TARGET_OBJECTS:kflib_object>)
add_library(kflib_shared SHARED $<TARGET_OBJECTS:kflib_object>)


