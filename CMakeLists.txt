cmake_minimum_required(VERSION 3.21)

project("DeepTreeEcho-RWKV" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# Build type
if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Deep Tree Echo RWKV.cpp Integration
option(ECHO_BUILD_RWKV_CPP "Build rwkv.cpp integration" ON)

if(ECHO_BUILD_RWKV_CPP)
    # Disable tests for rwkv.cpp to avoid build issues
    set(RWKV_BUILD_TESTS OFF CACHE BOOL "Build rwkv.cpp tests" FORCE)
    
    # Add rwkv.cpp as subdirectory
    add_subdirectory(external/rwkv.cpp)
    
    # Create shared library target for Deep Tree Echo integration
    add_library(echo_rwkv_cpp_bridge SHARED
        src/rwkv_cpp_bridge.cpp
    )
    
    # Link with rwkv library
    target_link_libraries(echo_rwkv_cpp_bridge
        rwkv
    )
    
    # Include directories
    target_include_directories(echo_rwkv_cpp_bridge PRIVATE
        external/rwkv.cpp
        src
    )
    
    # Set output directory for Python to find the shared library
    set_target_properties(echo_rwkv_cpp_bridge PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    
    # Platform-specific settings
    if(WIN32)
        set_target_properties(echo_rwkv_cpp_bridge PROPERTIES
            SUFFIX ".dll"
            PREFIX ""
        )
    elseif(APPLE)
        set_target_properties(echo_rwkv_cpp_bridge PROPERTIES
            SUFFIX ".so"
            PREFIX "lib"
        )
    else()
        set_target_properties(echo_rwkv_cpp_bridge PROPERTIES
            SUFFIX ".so" 
            PREFIX "lib"
        )
    endif()
endif()

# Add build script target
add_custom_target(build_echo_rwkv
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target echo_rwkv_cpp_bridge
    COMMENT "Building Deep Tree Echo RWKV.cpp integration"
)