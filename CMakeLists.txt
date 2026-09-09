cmake_minimum_required(VERSION 3.10...3.29)

# Set extension name here
set(TARGET_NAME duckdo)

set(EXTENSION_NAME ${TARGET_NAME}_extension)
set(LOADABLE_EXTENSION_NAME ${TARGET_NAME}_loadable_extension)

project(${TARGET_NAME})

set(CMAKE_CXX_STANDARD "17" CACHE STRING "C++ standard to enforce")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(src/include)

set(EXTENSION_SOURCES
    src/duckdo_extension.cpp
    src/linalg.cpp
    src/frame.cpp
    src/estimators.cpp
    src/functions.cpp
    src/diagnostics.cpp
    src/graph.cpp
    src/interventions.cpp
    src/runtime.cpp
    src/continuous.cpp
    src/panel.cpp
    src/identification.cpp
    src/mediation.cpp
    src/longitudinal.cpp
    src/survival.cpp
    src/frame_summary.cpp)

# Causal foundation models are an opt-in build. Without them the extension still
# builds and runs with zero external dependencies, which is what keeps the
# community build's first impression download-free. See cmake/ort.cmake for how
# the runtime is acquired and why it is linked shared rather than static.
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/ort.cmake)
if(DUCKDO_WITH_ONNX)
  add_definitions(-DDUCKDO_WITH_ONNX)
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

if(DUCKDO_WITH_ONNX)
  # The ONNX Runtime C++ headers need C++17: noexcept in typedefs and
  # structured bindings. DuckDB's build settles on an older standard, so raise
  # it for our targets only.
  foreach(duckdo_target ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    set_property(TARGET ${duckdo_target} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${duckdo_target} PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(${duckdo_target} duckdo_onnxruntime)
  endforeach()

  # Windows ships its own C:\Windows\System32\onnxruntime.dll with Windows ML,
  # and it is old enough to hand back a null OrtApi for the ORT_API_VERSION this
  # extension compiles against - which crashes on the first inference rather
  # than failing to load. The loader searches the host executable's directory
  # before System32, so staging our copy next to the shell and the test binary
  # is what makes ours win. This used to be a manual copy; a build step that
  # people forget is a build step that does not exist.
  if(WIN32 AND DUCKDO_ORT_LIB_DIR)
    file(GLOB _duckdo_ort_dlls "${DUCKDO_ORT_LIB_DIR}/*.dll")
    if(_duckdo_ort_dlls)
      add_custom_command(TARGET ${LOADABLE_EXTENSION_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/test"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_duckdo_ort_dlls} "${CMAKE_BINARY_DIR}/test/"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_duckdo_ort_dlls} "${CMAKE_BINARY_DIR}/"
        COMMENT "DuckDo: staging ONNX Runtime DLL(s) ahead of the System32 copy")
    endif()
  endif()
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
