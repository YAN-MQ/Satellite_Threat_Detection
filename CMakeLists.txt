build_exec(
  EXECNAME realtime_satellite
  SOURCE_FILES realtime_satellite.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
    ${libtap-bridge}
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/06_realtime_emulation
)

build_exec(
  EXECNAME federated_constellation
  SOURCE_FILES federated_constellation.cc
  LIBRARIES_TO_LINK
    ${libcore}
    ${libnetwork}
    ${libinternet}
    ${libpoint-to-point}
  EXECUTABLE_DIRECTORY_PATH ${CMAKE_OUTPUT_DIRECTORY}/scratch/06_realtime_emulation
)
