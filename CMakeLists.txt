# CMakeList.txt : CMake project for ObjectRecognition, include source and define
# project specific logic here.
#

# Add source to this project's executable.
add_executable (ObjectRecognition "processing.h" "main.h" "processing.cpp" "csv_util.h"  "csv_util.cpp" "knn.cpp" "knn.h")

if (CMAKE_VERSION VERSION_GREATER 3.12)
  set_property(TARGET ObjectRecognition PROPERTY CXX_STANDARD 20)
endif()

# TODO: Add tests and install targets if needed.
