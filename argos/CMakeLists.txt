#
# Loop functions
#
add_library(collectiveRlTransport SHARED
  collectiveRlTransport.h
  collectiveRlTransport_qtuser_functions.h
  collectiveRlTransport.cpp
  collectiveRlTransport_qtuser_functions.cpp)
  
target_link_libraries(collectiveRlTransport
  ${BUZZ_LIBRARY}
  ${ZeroMQ_LINK_LIBRARIES}
  argos3core_simulator
  argos3plugin_simulator_dynamics2d
  argos3plugin_simulator_entities
  argos3plugin_simulator_footbot
  argos3plugin_simulator_genericrobot
  argos3plugin_simulator_media
  argos3plugin_simulator_qtopengl
  argos3plugin_simulator_buzz)

#
# Buzz code
#
include(UseBuzz)
buzz_make(collectiveRlTransport.bzz)
