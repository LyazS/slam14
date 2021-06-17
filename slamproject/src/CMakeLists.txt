# 生成一个库文件
# 搜索当前目录下的所有.cpp文件
aux_source_directory(. SRC_CPP)
add_library(myslam ${SRC_CPP})
target_link_libraries(myslam ${THIRD_PARTY_LIBS} fmt)