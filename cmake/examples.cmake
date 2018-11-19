LINK_DIRECTORIES($ENV{HOME}/local/openblas/lib) # FIXME: find

FUNCTION(ADD_NN_OPS_EXAMPLE target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_INCLUDE_DIRECTORIES(${target} PRIVATE ${CMAKE_SOURCE_DIR}/include)
    ADD_DEPENDENCIES(${target} libstdtensor)
    IF(USE_OPENBLAS)
        TARGET_LINK_LIBRARIES(${target} openblas)
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

FILE(GLOB examples examples/example_*.cpp)
FOREACH(f ${examples})
    GET_FILENAME_COMPONENT(name ${f} NAME_WE)
    STRING(REPLACE "_"
                   "-"
                   name
                   ${name})
    ADD_NN_OPS_EXAMPLE(${name} ${f})
ENDFOREACH()
