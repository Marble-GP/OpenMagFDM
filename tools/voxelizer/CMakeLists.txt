cmake_minimum_required(VERSION 3.16)
project(OpenMagVoxelizer VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
# Standalone build: no dependency on the 2D solver, OpenCV, or VTK.
find_package(OpenCASCADE REQUIRED COMPONENTS
    FoundationClasses ModelingData ModelingAlgorithms Visualization
    ApplicationFramework DataExchange)
if(OpenCASCADE_MAJOR_VERSION LESS 7 OR
   (OpenCASCADE_MAJOR_VERSION EQUAL 7 AND OpenCASCADE_MINOR_VERSION LESS 6))
    message(FATAL_ERROR "Open CASCADE 7.6 or later is required")
endif()
if(TARGET TKDESTEP)
    set(OMFDM_STEP_LIBRARY TKDESTEP)
else()
    set(OMFDM_STEP_LIBRARY TKSTEP)
endif()

add_library(omfdm_voxel_core STATIC voxelizer.cpp vtk_writer.cpp)
target_compile_features(omfdm_voxel_core PUBLIC cxx_std_17)
target_include_directories(omfdm_voxel_core SYSTEM PUBLIC ${OpenCASCADE_INCLUDE_DIR})
target_include_directories(omfdm_voxel_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_link_libraries(omfdm_voxel_core PUBLIC ${OMFDM_STEP_LIBRARY}
    TKXSBase TKPrim TKTopAlgo TKBRep TKGeomBase TKG3d TKG2d TKMath TKernel)
if(MSVC)
    target_compile_options(omfdm_voxel_core PRIVATE /W4 /utf-8 /fp:precise)
else()
    target_compile_options(omfdm_voxel_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(MagVoxelizer main.cpp)
target_link_libraries(MagVoxelizer PRIVATE omfdm_voxel_core)
add_executable(MagCadPreview cad_preview.cpp)
target_link_libraries(MagCadPreview PRIVATE omfdm_voxel_core TKMesh)
if(MSVC)
    target_compile_options(MagVoxelizer PRIVATE /utf-8)
    target_compile_options(MagCadPreview PRIVATE /utf-8)
endif()
install(TARGETS MagVoxelizer MagCadPreview RUNTIME DESTINATION bin)

if(BUILD_TESTING)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    add_executable(voxelizer_geometry_test tests/geometry_test.cpp)
    target_link_libraries(voxelizer_geometry_test PRIVATE omfdm_voxel_core TKBO TKBool)
    add_test(NAME voxelizer_geometry COMMAND voxelizer_geometry_test
        ${CMAKE_CURRENT_BINARY_DIR}/geometry_fixtures)
    add_test(NAME voxelizer_cli COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.py
        $<TARGET_FILE:MagVoxelizer> $<TARGET_FILE:voxelizer_geometry_test>
        ${CMAKE_CURRENT_BINARY_DIR}/cli_fixtures)
    option(OMFDM_TEST_VOXEL_VIEWER "Test the optional Python/VTK viewer (needs OpenGL)" OFF)
    if(OMFDM_TEST_VOXEL_VIEWER)
        add_test(NAME voxelizer_viewer COMMAND ${Python3_EXECUTABLE}
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/viewer_test.py)
    endif()
endif()
