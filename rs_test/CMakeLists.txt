cmake_minimum_required(VERSION 3.2)
project(test)

set(ASMJIT_STATIC ON CACHE BOOL "" FORCE)
add_subdirectory(asmjit EXCLUDE_FROM_ALL)

set(SOURCE_FILES test_rs.cpp)
add_executable(test ${SOURCE_FILES})
target_link_libraries(test asmjit::asmjit)