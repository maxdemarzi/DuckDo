# ONNX Runtime acquisition for the DuckDo extension.
#
# The classical estimators have no external dependency at all, and that is worth
# protecting: a build with no flags produces an extension that does causal
# inference out of the box. The causal foundation models need ONNX Runtime, so
# they are opt-in:
#
#   -DDUCKDO_WITH_ONNX=ON            fetch the official prebuilt release and
#                                    build the model path
#   -DDUCKDO_ONNXRUNTIME_ROOT=<dir>  use an already-unpacked release instead
#                                    (implies DUCKDO_WITH_ONNX)
#   -DDUCKDO_ORT_URL=<url>           mirror, for networks where GitHub releases
#                                    are unreachable
#
# The roadmap asked for a *static* link here, on the theory that a community
# build cannot ship a shared dependency. That turned out to be wrong: no
# official static build of ONNX Runtime is published, building one from source
# is a multi-hour per-platform job, and the extension this project took as its
# precedent (anofox_tabfm) links the shared library and stages it next to the
# binaries. This does the same.
#
# Outputs: the duckdo_onnxruntime INTERFACE target, and DUCKDO_ORT_LIB_DIR for
# the staging step in CMakeLists.txt.

option(DUCKDO_WITH_ONNX "Build the causal foundation model path (fetches ONNX Runtime)" OFF)
set(DUCKDO_ORT_VERSION "1.29.0" CACHE STRING "ONNX Runtime version to fetch")
set(DUCKDO_ORT_URL "" CACHE STRING "Override URL for the prebuilt ONNX Runtime archive (mirror support)")
set(DUCKDO_ONNXRUNTIME_ROOT "" CACHE PATH "Root of an already-unpacked ONNX Runtime release")

# Pointing at a root is an unambiguous request for the model path.
if(DUCKDO_ONNXRUNTIME_ROOT)
  set(DUCKDO_WITH_ONNX ON)
endif()

if(NOT DUCKDO_WITH_ONNX)
  message(STATUS "DuckDo: building without ONNX Runtime (classical estimators only). "
                 "Pass -DDUCKDO_WITH_ONNX=ON for the foundation-model path.")
  return()
endif()

# IMPORTED so the target may be referenced by the exported extension targets:
# install(EXPORT DuckDBExports) rejects non-imported build-tree targets.
add_library(duckdo_onnxruntime INTERFACE IMPORTED GLOBAL)

# Map the target platform to the release archive's name.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_duckdo_ort_platform "linux-aarch64")
  else()
    set(_duckdo_ort_platform "linux-x64")
  endif()
  set(_duckdo_ort_ext "tgz")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
  # DuckDB's extension matrix cross-builds osx_amd64 on an arm64 runner, so the
  # host arch says nothing about the target. The universal2 archive carries both.
  set(_duckdo_ort_platform "osx-universal2")
  set(_duckdo_ort_ext "tgz")
elseif(WIN32)
  set(_duckdo_ort_platform "win-x64")
  set(_duckdo_ort_ext "zip")
else()
  message(FATAL_ERROR "DuckDo: no prebuilt ONNX Runtime is mapped for ${CMAKE_SYSTEM_NAME}. "
                      "Unpack one yourself and pass -DDUCKDO_ONNXRUNTIME_ROOT.")
endif()

if(DUCKDO_ONNXRUNTIME_ROOT)
  message(STATUS "DuckDo: ONNX Runtime from ${DUCKDO_ONNXRUNTIME_ROOT}")
  set(_duckdo_ort_root "${DUCKDO_ONNXRUNTIME_ROOT}")
else()
  if(DUCKDO_ORT_URL)
    set(_duckdo_ort_archive "${DUCKDO_ORT_URL}")
  else()
    set(_duckdo_ort_archive
        "https://github.com/microsoft/onnxruntime/releases/download/v${DUCKDO_ORT_VERSION}/onnxruntime-${_duckdo_ort_platform}-${DUCKDO_ORT_VERSION}.${_duckdo_ort_ext}")
  endif()
  message(STATUS "DuckDo: fetching ONNX Runtime v${DUCKDO_ORT_VERSION} (${_duckdo_ort_platform})")
  include(FetchContent)
  FetchContent_Declare(duckdo_ort_prebuilt URL "${_duckdo_ort_archive}")
  FetchContent_MakeAvailable(duckdo_ort_prebuilt)
  set(_duckdo_ort_root "${duckdo_ort_prebuilt_SOURCE_DIR}")
endif()

find_library(DUCKDO_ORT_LIB
             NAMES onnxruntime
             PATHS "${_duckdo_ort_root}/lib"
             NO_DEFAULT_PATH REQUIRED)
target_include_directories(duckdo_onnxruntime INTERFACE "${_duckdo_ort_root}/include")
target_link_libraries(duckdo_onnxruntime INTERFACE "${DUCKDO_ORT_LIB}")

set(DUCKDO_ORT_LIB_DIR "${_duckdo_ort_root}/lib")
