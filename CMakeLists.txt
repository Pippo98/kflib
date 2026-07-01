cmake_minimum_required(VERSION 3.20)
project(KF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DIR ${CMAKE_CURRENT_LIST_DIR})

if(NOT TARGET Eigen3::Eigen)
  add_subdirectory(${DIR}/eigen)
  message(STATUS "Eigen3::Eigen added from ${DIR}/eigen")
endif()

add_library(kflib_object
  OBJECT
  ${DIR}/src/kf_base.cpp
  ${DIR}/src/kf.cpp
  ${DIR}/src/ekf.cpp
  ${DIR}/src/ukf.cpp
)
set_target_properties(kflib_object PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(kflib_object PUBLIC Eigen3::Eigen)

add_library(kflib_static STATIC $<TARGET_OBJECTS:kflib_object>)
add_library(kflib_shared SHARED $<TARGET_OBJECTS:kflib_object>)

# include directories
target_include_directories(kflib_object PUBLIC ${DIR}/include)
target_include_directories(kflib_static PUBLIC ${DIR}/include)
target_include_directories(kflib_shared PUBLIC ${DIR}/include)
# link libraries
target_link_libraries(kflib_static PUBLIC Eigen3::Eigen)
target_link_libraries(kflib_shared PUBLIC Eigen3::Eigen)

target_precompile_headers(kflib_static PRIVATE <Eigen/Core>)
target_precompile_headers(kflib_shared PRIVATE <Eigen/Core>)

if(KFLIB_BUILD_EXAMPLES)
  add_executable(example_kf ${DIR}/examples/example_kf.cpp)
  target_link_libraries(example_kf PRIVATE kflib_static)
  add_executable(example_ekf ${DIR}/examples/example_ekf.cpp)
  target_link_libraries(example_ekf PRIVATE kflib_static)
  add_executable(example_ukf ${DIR}/examples/example_ukf.cpp)
  target_link_libraries(example_ukf PRIVATE kflib_static)
endif()
