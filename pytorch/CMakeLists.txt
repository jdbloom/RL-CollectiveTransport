add_library(pytorchcomm SHARED pytorchcomm.h pytorchcomm.cpp)
target_link_libraries(pytorchcomm ${ZeroMQ_LINK_LIBRARIES})

## create an executable from InterfaceTest.cpp
#add_executable(testServerClient InterfaceTest.cpp)
#target_link_libraries(testServerClient serverclient)

## create a sepereate executable
#add_executable(testServerClient2 InterfaceTest.cpp)
#target_link_libraries(testServerClient2 serverclient)
