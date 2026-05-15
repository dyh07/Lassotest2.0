cmake_minimum_required(VERSION 3.21)
project(atlas_lens_qt VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

# --- Find Qt6 ---
find_package(Qt6 REQUIRED COMPONENTS Core Gui Widgets WebEngineWidgets Network Concurrent)

# --- Find HDF5 (for HighFive) ---
# Try to find via CMake first; fall back to conda environment
find_package(HDF5 QUIET)
if(NOT HDF5_FOUND)
    # Try conda environment
    if(DEFINED ENV{CONDA_PREFIX})
        set(HDF5_ROOT "$ENV{CONDA_PREFIX}")
        find_package(HDF5 REQUIRED)
    endif()
endif()
# If still not found, try known conda path
if(NOT HDF5_FOUND)
    if(EXISTS "D:/conda/Library/include/hdf5.h")
        set(HDF5_FOUND TRUE)
        set(HDF5_INCLUDE_DIRS "D:/conda/Library/include")
        set(HDF5_LIBRARIES "D:/conda/Library/lib/hdf5.lib")
    endif()
endif()

# --- Third-party headers (fetched or bundled) ---
set(THIRD_PARTY_DIR "${CMAKE_SOURCE_DIR}/third_party")

# Crow (header-only HTTP framework)
set(CROW_INCLUDE_DIR "${THIRD_PARTY_DIR}/Crow/include")

# nlohmann/json (header-only JSON)
set(NLOHMANN_JSON_INCLUDE_DIR "${THIRD_PARTY_DIR}/json/single_include")

# HighFive (header-only HDF5 wrapper)
set(HIGHFIVE_INCLUDE_DIR "${THIRD_PARTY_DIR}/HighFive/include")

# Asio (required by Crow, standalone mode)
if(EXISTS "D:/conda/Library/include/asio.hpp")
    set(ASIO_INCLUDE_DIR "D:/conda/Library/include")
else()
    set(ASIO_INCLUDE_DIR "${THIRD_PARTY_DIR}/asio/include")
endif()

# --- Source files ---
set(SOURCES
    src/main.cpp
    src/http_server.cpp
    src/h5ad_reader.cpp
    src/data_store.cpp
    src/lasso_view.cpp
    src/downsample.cpp
    src/analysis_runner.cpp
)

set(HEADERS
    src/http_server.h
    src/h5ad_reader.h
    src/data_store.h
    src/ann_data.h
    src/lasso_view.h
    src/downsample.h
    src/analysis_runner.h
    src/json_utils.h
)

# --- Qt resource file for embedding frontend ---
qt6_add_resources(RESOURCES resources.qrc)

# --- Executable ---
add_executable(${PROJECT_NAME} WIN32 ${SOURCES} ${HEADERS} ${RESOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CROW_INCLUDE_DIR}
    ${NLOHMANN_JSON_INCLUDE_DIR}
    ${HIGHFIVE_INCLUDE_DIR}
    ${ASIO_INCLUDE_DIR}
    ${HDF5_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    Qt6::WebEngineWidgets
    Qt6::Network
    Qt6::Concurrent
    ${HDF5_LIBRARIES}
)

if(WIN32)
    target_compile_definitions(${PROJECT_NAME} PRIVATE
        CROW_STATIC_DEFINITION
        CROW_MSVC_WORKAROUND
        CROW_DISABLE_STATIC_DIR
        NOMINMAX
        _USE_MATH_DEFINES
        _WIN32_WINNT=0x0601
        ASIO_STANDALONE
    )
    # On Windows, link ws2_32 for Crow (socket support)
    target_link_libraries(${PROJECT_NAME} PRIVATE ws2_32)
endif()

# --- Copy frontend files to build directory ---
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/frontend"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/frontend"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/sample_data"
        "$<TARGET_FILE_DIR:${PROJECT_NAME}>/sample_data"
)
