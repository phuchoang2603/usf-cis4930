cmake_minimum_required(VERSION 3.16)
project(project_evaluator LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT VNRC_SOURCE OR NOT JSON_INCLUDE)
  message(FATAL_ERROR "Set VNRC_SOURCE and JSON_INCLUDE; use scripts/setup_vnrc.py")
endif()
add_library(evaluator_backend SHARED bridge.cpp
  "${VNRC_SOURCE}/src/core/subgraph.cpp"
  "${VNRC_SOURCE}/src/core/dag.cpp"
  "${VNRC_SOURCE}/src/core/cost.cpp")
target_include_directories(evaluator_backend PRIVATE "${VNRC_SOURCE}/src" "${JSON_INCLUDE}")
target_compile_options(evaluator_backend PRIVATE -O2 -UNDEBUG)
