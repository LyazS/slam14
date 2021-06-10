cmake_minimum_required(VERSION 3.0.0)
project(SlamProject VERSION 0.1.0)

include(CTest)
enable_testing()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)


find_package(Sophus REQUIRED)
find_package(OpenCV REQUIRED)
set(THIRD_PARTY_LIBS ${OpenCV_LIBS} ${Sophus_LIBRARIES})

# 包含头文件
include_directories({OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories("./include/")
# 添加子目录，子目录下也需要CMakeLists.txt
# 这样子可以编译生成一个库文件
add_subdirectory(src)
add_subdirectory(test)


aux_source_directory(. SRC_LIST)
add_executable(SlamProject ${SRC_LIST})
target_link_libraries(SlamProject ${THIRD_PARTY_LIBS} fmt)