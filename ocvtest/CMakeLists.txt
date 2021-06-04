cmake_minimum_required(VERSION 3.0.0)
project(hello VERSION 0.1.0)

include(CTest)
enable_testing()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# find OpenCV
find_package(OpenCV REQUIRED)

# show the message of OpenCV
message(STATUS "OpenCV library status:")
message(STATUS "    version: 	${OpenCV_VERSION}")
message(STATUS "    headers: 	${OpenCV_INCLUDE_DIRS}")
message(STATUS "    libraries: 	${OpenCV_LIBS}")

# link headers
include_directories({OpenCV_INCLUDE_DIRS})
include_directories("/usr/include/eigen3")
include_directories("./")
# 添加源代码文件到SRC_LIST变量中
aux_source_directory(. SRC_LIST)

# 生成可执行文件
add_executable(hello ${SRC_LIST})

# after ADD_EXECUTABLE，为生成文件target添加库
target_link_libraries(hello ${OpenCV_LIBS})