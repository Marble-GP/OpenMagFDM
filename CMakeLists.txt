cmake_minimum_required(VERSION 3.5)

project(MagFDMsolver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(MagFDMsolver
    MagFDMsolver.cpp
    MagneticFieldAnalyzer.cpp
    MagneticFieldAnalyzer_nonlinear.cpp
    MagneticFieldAnalyzer_nonlinear_newton.cpp
    MagneticFieldAnalyzer.h
    tinyexpr/tinyexpr.cpp
    tinyexpr/tinyexpr.h
)

include(GNUInstallDirs)
install(TARGETS MagFDMsolver
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")

# ライブラリを探す
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(yaml-cpp REQUIRED)

# OpenMP (optional: single-threaded build if not available)
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION} — parallel build enabled")
else()
    message(STATUS "OpenMP not found — single-threaded build")
endif()

# AMGCL (header-only library)
include_directories(${CMAKE_SOURCE_DIR}/amgcl)
add_definitions(-DAMGCL_NO_BOOST)

# tinyexpr (header-only library)
include_directories(${CMAKE_SOURCE_DIR}/tinyexpr)

# 見つかったライブラリの情報を表示（デバッグ用）
message(STATUS "Eigen3 version: ${Eigen3_VERSION}")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "yaml-cpp version: ${yaml-cpp_VERSION}")

# ライブラリのリンク
# yaml-cpp::yaml-cpp はモダンCMakeターゲットで、インクルードパスも自動設定される
target_link_libraries(MagFDMsolver
    Eigen3::Eigen
    ${OpenCV_LIBS}
    yaml-cpp::yaml-cpp
    $<$<BOOL:${OpenMP_CXX_FOUND}>:OpenMP::OpenMP_CXX>
)
