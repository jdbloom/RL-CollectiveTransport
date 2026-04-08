#
# Loop functions
#
set(LIBRARY_FILES collectiveRlTransport.h collectiveRlTransport.cpp)

if(NOT CLUSTER)
    set(LIBRARY_FILES ${LIBRARY_FILES} collectiveRlTransport_qtuser_functions.h collectiveRlTransport_qtuser_functions.cpp)
endif(NOT CLUSTER)

add_library(collectiveRlTransport SHARED ${LIBRARY_FILES})

  target_link_libraries(collectiveRlTransport
  ${BUZZ_LIBRARY}
    ${ZeroMQ_LINK_LIBRARIES}
    argos3core_simulator
    argos3plugin_simulator_dynamics2d
    argos3plugin_simulator_entities
    argos3plugin_simulator_footbot
    argos3plugin_simulator_genericrobot
    argos3plugin_simulator_media
    argos3plugin_simulator_buzz
    argos3plugin_simulator_nonuniform_objects)
  if(NOT CLUSTER)
    target_link_libraries(collectiveRlTransport
      argos3plugin_simulator_qtopengl)
  endif(NOT CLUSTER)

#
# Buzz code
#

buzz_make(collectiveRlTransport.bzz)
