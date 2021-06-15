cmake_minimum_required(VERSION 3.0.0)
project(hello VERSION 0.1.0)

include(CTest)
enable_testing()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

# g2o
LIST( APPEND CMAKE_MODULE_PATH /home/install_thing/g2o_old/g2o-20170730_git/cmake_modules )
SET( G2O_ROOT /home/install_thing/g2o_old/g2o-20170730_git )
find_package(G2O REQUIRED)


find_package(Sophus REQUIRED)
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
include_directories(${G2O_INCLUDE_DIRS})
include_directories("./")
# 添加源代码文件到SRC_LIST变量中
aux_source_directory(. SRC_LIST)

# 生成可执行文件
add_executable(hello ${SRC_LIST})

# after ADD_EXECUTABLE，为生成文件target添加库
target_link_libraries(hello ${OpenCV_LIBS})
target_link_libraries(hello ${Sophus_LIBRARIES} fmt)
target_link_libraries(hello ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})