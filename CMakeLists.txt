cmake_minimum_required(VERSION 3.10)
project(LoadSavedModel)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# TensorFlow paths - update these for your system
set(TF_SOURCE_DIR "/data/jinying/tensorflow")
set(TF_PYTHON_INCLUDE "/data/jinying/miniconda/envs/tfexport/lib/python3.8/site-packages/tensorflow/include")
set(TF_LIB_DIR "/data/jinying/jacrev1/tensorflow-cpp/lib")
set(ML_DTYPES_INCLUDE "/data/jinying/jacrev2/third_party/ml_dtypes/include/ml_dtypes/include")
set(TSL_INCLUDE "/data/jinying/tensorflow/third_party/xla/third_party/tsl/tsl")

# Include paths
include_directories(
    ${TF_SOURCE_DIR}
    ${TF_PYTHON_INCLUDE}
    ${TF_PYTHON_INCLUDE}/tensorflow
    ${ML_DTYPES_INCLUDE}
    ${TSL_INCLUDE}
)

# Link directories
link_directories(${TF_LIB_DIR})

# Build executable
add_executable(main main.cpp)

# Link libraries
target_link_libraries(main
    tensorflow
    tensorflow_framework
    pthread
    dl
)

# Set RPATH for runtime library search
set_target_properties(main PROPERTIES
    BUILD_RPATH "${TF_LIB_DIR}"
)

# Useful targets
add_custom_target(run
    COMMAND ./main 2>/dev/null || ./main
    DEPENDS main
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)