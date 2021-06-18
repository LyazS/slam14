# 生成一个库文件
# 搜索当前目录下的所有.cpp文件
aux_source_directory(. SRC_CPP)
add_library(myslam ${SRC_CPP})
target_link_libraries(myslam ${THIRD_PARTY_LIBS} fmt)
target_link_libraries(myslam ${CSPARSE_LIBRARY})
target_link_libraries(myslam 
g2o_core 
g2o_stuff 
g2o_types_sba  
g2o_types_slam3d 
g2o_solver_csparse 
g2o_csparse_extension
cholmod 
g2o_solver_cholmod)