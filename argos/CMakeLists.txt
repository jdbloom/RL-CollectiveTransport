#
# Loop functions
#

message("VICON is set to : ${VICON}")
if(VICON)
    message("collectiveRLTransportNetworking.cpp is being selected to run")
    set(LIBRARY_FILES collectiveRlTransportNetworking.h collectiveRlTransportNetworking.cpp)
else()
    message("collectiveRLTransport.cpp is being selected to run")
    set(LIBRARY_FILES collectiveRlTransport.h collectiveRlTransport.cpp)
endif(VICON)

if(NOT CLUSTER)
    set(LIBRARY_FILES ${LIBRARY_FILES} collectiveRlTransport_qtuser_functions.h collectiveRlTransport_qtuser_functions.cpp)
endif(NOT CLUSTER)

add_library(collectiveRlTransport SHARED ${LIBRARY_FILES})

#
# Let it be known, footbot was removed from below and is now kheperaiv
#
  target_link_libraries(collectiveRlTransport
  ${BUZZ_LIBRARY}
    ${ZeroMQ_LINK_LIBRARIES}
    argos3core_simulator
    argos3plugin_simulator_dynamics2d
    argos3plugin_simulator_entities
    argos3plugin_simulator_kheperaiv
    argos3plugin_simulator_genericrobot
    argos3plugin_simulator_media
    argos3plugin_simulator_buzz)
  if(NOT CLUSTER)
    target_link_libraries(collectiveRlTransport
      argos3plugin_simulator_qtopengl)
  endif(NOT CLUSTER)
  if(VICON)
    target_link_libraries(collectiveRlTransport
      argos3plugins_simulator_moveable_tcp_socket
      argos3plugins_simulator_tracking)
  endif(VICON)

#
# Buzz code
#

buzz_make(collectiveRlTransport.bzz)
