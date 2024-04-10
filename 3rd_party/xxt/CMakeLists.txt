cmake_minimum_required(VERSION 3.22)
project(xxt VERSION 0.0.1 DESCRIPTION
  "XXT: A highly scalable sparse direct solver" LANGUAGES C CXX)

# Build options:
option(ENABLE_BLAS "Enable BLAS support" OFF)

set(XXT_SOURCES src/xxt.cpp)

add_library(xxt SHARED ${XXT_SOURCES})
set_target_properties(xxt PROPERTIES
  VERSION ${PROJECT_VERSION}
  SOVERSION 1
  PUBLIC_HEADER include/xxt.hpp)
target_include_directories(xxt PRIVATE include)
target_compile_options(xxt PRIVATE $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Wno-unused-function
  -Wno-unused-parameter -Wno-stringop-overflow>)
target_link_libraries(xxt PRIVATE gs)

if (ENABLE_BLAS)
  find_package(LAPACK REQUIRED)
  find_package(BLAS REQUIRED)
  target_compile_definitions(xxt PRIVATE ENABLE_BLAS)
  target_link_libraries(xxt PRIVATE LAPACK::LAPACK BLAS::BLAS)
endif()

install(TARGETS xxt LIBRARY DESTINATION lib PUBLIC_HEADER DESTINATION include)

# Add clang-format as a custom target if available.
find_program(CLANG_FORMAT NAMES clang-format)
if (CLANG_FORMAT)
  set(FILES_TO_FORMAT ${CMAKE_SOURCE_DIR}/src/*.c ${CMAKE_SOURCE_DIR}/backends/*.c ${CMAKE_SOURCE_DIR}/bin/*.c)
  set(FILES_TO_FORMAT ${FILES_TO_FORMAT} ${CMAKE_SOURCE_DIR}/include/*.h ${CMAKE_SOURCE_DIR}/include/backends/*.h)
  add_custom_target(format-xxt
    COMMAND ${CLANG_FORMAT} -i ${FILES_TO_FORMAT}
    COMMENT "Running clang-format ...")
  add_custom_target(format-check-xxt
    COMMAND ${CLANG_FORMAT} --dry-run -Werror -i ${FILES_TO_FORMAT}
    COMMENT "Running clang-format check ...")
endif()

# Configure .clang-tidy.in to generate .clang-tidy.
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy.in
  ${CMAKE_CURRENT_SOURCE_DIR}/.clang-tidy @ONLY)
# Add clang-tidy as a custom target if available.
find_program(CLANG_TIDY NAMES clang-tidy)
if (CLANG_TIDY)
  add_custom_target(tidy-xxt
    COMMAND ${CLANG_TIDY} -p ${CMAKE_CURRENT_BINARY_DIR}
      ${CMAKE_CURRENT_SOURCE_DIR}/*/*.[ch]
    COMMENT "Running clang-tidy ...")
endif()
