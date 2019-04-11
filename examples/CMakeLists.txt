IF(TRACE_EXAMPLES)
    FIND_PACKAGE(stdtracer REQUIRED)
ENDIF()

FUNCTION(ADD_NN_OPS_EXAMPLE target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    ADD_DEPENDENCIES(${target} libstdtensor)
    IF(TRACE_EXAMPLES)
        ADD_DEPENDENCIES(${target} libstdtracer)
        TARGET_SOURCES(${target} PRIVATE trace.cpp)
    ENDIF()
    IF(USE_OPENBLAS)
        TARGET_LINK_LIBRARIES(${target} ${OpenBLAS_LIBRARIES})
    ENDIF()
    IF(USE_OPENCV)
        TARGET_COMPILE_DEFINITIONS(${target} PRIVATE USE_OPENCV)
        TARGET_LINK_LIBRARIES(${target}
                              opencv_core
                              opencv_imgproc
                              opencv_highgui
                              opencv_imgcodecs)
    ENDIF()
ENDFUNCTION()

FILE(GLOB examples example_*.cpp)
FOREACH(f ${examples})
    GET_FILENAME_COMPONENT(name ${f} NAME_WE)
    STRING(REPLACE "_"
                   "-"
                   name
                   ${name})
    ADD_NN_OPS_EXAMPLE(${name} ${f})
ENDFOREACH()
