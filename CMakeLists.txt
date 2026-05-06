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

option(ORBITSHIELD_ENABLE_LIBTORCH_RUNTIME "Enable the experimental libtorch Level 4B runtime target" ON)

if(ORBITSHIELD_ENABLE_LIBTORCH_RUNTIME)
  set(ORBITSHIELD_TORCH_CMAKE_PREFIX "/home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/share/cmake")
  if(EXISTS "${ORBITSHIELD_TORCH_CMAKE_PREFIX}/Torch/TorchConfig.cmake")
    set(CMAKE_PREFIX_PATH "${ORBITSHIELD_TORCH_CMAKE_PREFIX}" CACHE STRING "OrbitShield torch prefix" FORCE)
    set(Torch_DIR "${ORBITSHIELD_TORCH_CMAKE_PREFIX}/Torch" CACHE PATH "OrbitShield Torch config path" FORCE)
    set(Caffe2_DIR "${ORBITSHIELD_TORCH_CMAKE_PREFIX}/Caffe2" CACHE PATH "OrbitShield Caffe2 config path" FORCE)
    set(CMAKE_CUDA_STANDARD 17 CACHE STRING "OrbitShield libtorch CUDA standard" FORCE)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON CACHE BOOL "OrbitShield libtorch CUDA standard required" FORCE)
    set(CMAKE_CUDA_EXTENSIONS ON CACHE BOOL "OrbitShield libtorch CUDA extensions" FORCE)
    if(EXISTS "/usr/local/cuda/bin/nvcc")
      set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc" CACHE FILEPATH "OrbitShield libtorch CUDA compiler" FORCE)
    endif()
    set(CMAKE_CUDA_ARCHITECTURES "89" CACHE STRING "OrbitShield libtorch CUDA arch" FORCE)
    list(PREPEND CMAKE_PREFIX_PATH "${ORBITSHIELD_TORCH_CMAKE_PREFIX}")
    find_package(Torch QUIET)
  endif()

  if(TORCH_FOUND)
    add_executable(
      scratch_06_realtime_emulation_federated_libtorch_runtime
      federated_libtorch_runtime.cc
    )

    string(REPLACE " " ";" ORBITSHIELD_TORCH_CXX_FLAGS "${TORCH_CXX_FLAGS}")
    target_compile_options(
      scratch_06_realtime_emulation_federated_libtorch_runtime
      PRIVATE
        ${ORBITSHIELD_TORCH_CXX_FLAGS}
    )
    target_link_libraries(
      scratch_06_realtime_emulation_federated_libtorch_runtime
      PRIVATE
        ${LIB_AS_NEEDED_PRE}
        ${libcore}
        ${TORCH_LIBRARIES}
        ${LIB_AS_NEEDED_POST}
    )
    target_include_directories(
      scratch_06_realtime_emulation_federated_libtorch_runtime
      PRIVATE
        /home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/include
        /home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include
    )
    set_target_properties(
      scratch_06_realtime_emulation_federated_libtorch_runtime
      PROPERTIES
        CXX_STANDARD 23
        BUILD_RPATH "/home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/lib"
        INSTALL_RPATH "/home/lithic/final/ns3-gpu-venv/lib/python3.12/site-packages/torch/lib"
    )
    set_runtime_outputdirectory(
      "federated_libtorch_runtime"
      "${CMAKE_OUTPUT_DIRECTORY}/scratch/06_realtime_emulation/"
      "scratch_06_realtime_emulation_"
    )
  endif()
endif()
