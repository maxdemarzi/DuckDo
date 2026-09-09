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
    src/identification.cpp)

# Causal foundation models are an opt-in build: point DUCKDO_ONNXRUNTIME_ROOT at
# an unpacked onnxruntime release to enable them. Without it the extension still
# builds and runs with zero external dependencies, which is what keeps the
# community build's first impression download-free.
if(DUCKDO_ONNXRUNTIME_ROOT)
  message(STATUS "DuckDo: building with ONNX Runtime at ${DUCKDO_ONNXRUNTIME_ROOT}")
  add_definitions(-DDUCKDO_WITH_ONNX)
  include_directories(${DUCKDO_ONNXRUNTIME_ROOT}/include)
  find_library(DUCKDO_ORT_LIB
               NAMES onnxruntime
               PATHS ${DUCKDO_ONNXRUNTIME_ROOT}/lib
               NO_DEFAULT_PATH REQUIRED)
else()
  message(STATUS "DuckDo: building without ONNX Runtime (classical estimators only)")
endif()

build_static_extension(${TARGET_NAME} ${EXTENSION_SOURCES})
build_loadable_extension(${TARGET_NAME} " " ${EXTENSION_SOURCES})

if(DUCKDO_ONNXRUNTIME_ROOT)
  # The ONNX Runtime C++ headers need C++17: noexcept in typedefs and
  # structured bindings. DuckDB's build settles on an older standard, so raise
  # it for our targets only.
  foreach(duckdo_target ${EXTENSION_NAME} ${LOADABLE_EXTENSION_NAME})
    set_property(TARGET ${duckdo_target} PROPERTY CXX_STANDARD 17)
    set_property(TARGET ${duckdo_target} PROPERTY CXX_STANDARD_REQUIRED ON)
    target_link_libraries(${duckdo_target} ${DUCKDO_ORT_LIB})
  endforeach()
endif()

install(
  TARGETS ${EXTENSION_NAME}
  EXPORT "${DUCKDB_EXPORT_SET}"
  LIBRARY DESTINATION "${INSTALL_LIB_DIR}"
  ARCHIVE DESTINATION "${INSTALL_LIB_DIR}")
