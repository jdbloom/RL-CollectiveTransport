set(CMAKE_CXX_STANDARD 14)              # Enable c++14 standard

## use the hint from above to find where 'zmq.hpp' is located
find_path(ZeroMQ_INCLUDE_DIR
        NAMES zmq.hpp
        ##PATHS ${PC_ZeroMQ_INCLUDE_DIRS}
        )
message(STATUS "ZeroMQ_INCLUDE_DIR = ${ZeroMQ_INCLUDE_DIR}")
## use the hint from about to find the location of libzmq
find_library(ZeroMQ_LIBRARY
        NAMES zmq
        ##PATHS ${PC_ZeroMQ_LIBRARY_DIRS}
        )
message(STATUS "ZeroMQ_LIBRARY = ${ZeroMQ_LIBRARY}")

add_library(serverclient SHARED ModelServerClient.cpp ModelServerClient.hpp)

## add the include directory to our compile directives
target_include_directories(serverclient PUBLIC ${ZeroMQ_INCLUDE_DIR})
## at the 0mq library to our link directive
target_link_libraries(serverclient PUBLIC ${ZeroMQ_LIBRARY})

## create an executable from InterfaceTest.cpp
#add_executable(testServerClient InterfaceTest.cpp)
#target_link_libraries(testServerClient serverclient)

## create a sepereate executable
#add_executable(testServerClient2 InterfaceTest.cpp)
#target_link_libraries(testServerClient2 serverclient)
