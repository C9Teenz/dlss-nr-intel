cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED CMAKE_CXX_COMPILER AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tools/zig-cxx.cmd")
    set(CMAKE_CXX_COMPILER "${CMAKE_CURRENT_SOURCE_DIR}/tools/zig-cxx.cmd" CACHE FILEPATH "CXX compiler")
    set(CMAKE_AR "${CMAKE_CURRENT_SOURCE_DIR}/tools/zig-ar.cmd" CACHE FILEPATH "Archiver")
endif()

project(dlss_nr_dx12_poc LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/directml/extracted/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime/extracted/build/native/include
)

set(DX12_CORE_SOURCES
    src/dx12/device.cpp
    src/dx12/command_queue.cpp
    src/dx12/resources.cpp
    src/dx12/swapchain.cpp
    src/benchmark/benchmark.cpp
    src/bridge/d3d12_bridge.cpp
    src/neural/ort_runtime.cpp
    src/neural/neural_field.cpp
    src/neural/neural_graph.cpp
    src/neural/hybrid_ends.cpp
    src/neural/dml_gemm.cpp
    src/neural/neural_app.cpp
)

add_library(dlss_nr_dx12_core OBJECT ${DX12_CORE_SOURCES})

set(SYSTEM_DX12_LIBS
    d3d12
    dxgi
    dxguid
    d3dcompiler_47
    advapi32
    user32
    gdi32
    ole32
    uuid
)

add_executable(dlss_nr_dx12_poc src/app/main.cpp $<TARGET_OBJECTS:dlss_nr_dx12_core>)
target_link_libraries(dlss_nr_dx12_poc PRIVATE ${SYSTEM_DX12_LIBS})

# Drop-in Game Proxy DLL (dist/game-drop-in/dxgi.dll & DLSS_NR_DX12.asi)
add_library(dxgi_proxy SHARED
    src/bridge/dxgi_proxy.cpp
    src/bridge/dxgi.def
    src/bridge/d3d12_bridge.cpp
    src/neural/ort_runtime.cpp
    src/neural/neural_field.cpp
    src/neural/neural_graph.cpp
    src/neural/hybrid_ends.cpp
    src/neural/dml_gemm.cpp
    src/dx12/command_queue.cpp
    src/dx12/resources.cpp
)
set_target_properties(dxgi_proxy PROPERTIES
    OUTPUT_NAME "dxgi"
    PREFIX ""
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist/game-drop-in"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/dist/game-drop-in"
)
target_link_libraries(dxgi_proxy PRIVATE dxguid user32 ole32 uuid)

enable_testing()

add_executable(test_matmul tests/test_matmul.cpp $<TARGET_OBJECTS:dlss_nr_dx12_core>)
target_link_libraries(test_matmul PRIVATE ${SYSTEM_DX12_LIBS})
add_test(NAME test_matmul COMMAND test_matmul --shader-dir ${CMAKE_CURRENT_SOURCE_DIR}/src/compute)

add_executable(test_image tests/test_image.cpp $<TARGET_OBJECTS:dlss_nr_dx12_core>)
target_link_libraries(test_image PRIVATE ${SYSTEM_DX12_LIBS})
add_test(NAME test_image COMMAND test_image --shader-dir ${CMAKE_CURRENT_SOURCE_DIR}/src/compute --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/docs)

add_executable(test_bridge tests/test_bridge.cpp $<TARGET_OBJECTS:dlss_nr_dx12_core>)
target_link_libraries(test_bridge PRIVATE ${SYSTEM_DX12_LIBS})
add_test(NAME test_bridge COMMAND test_bridge --shader-dir ${CMAKE_CURRENT_SOURCE_DIR}/src/compute --output-dir ${CMAKE_CURRENT_SOURCE_DIR}/docs --frames 60)
