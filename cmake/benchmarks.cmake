INCLUDE(${CMAKE_SOURCE_DIR}/cmake/gbench.cmake)

ADD_CUSTOM_TARGET(benchmarks)

FIND_PACKAGE(Threads REQUIRED)
FUNCTION(ADD_BENCH target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_USE_GBENCH(${target})
    IF(UNIX)
        TARGET_LINK_LIBRARIES(${target} Threads::Threads)
    ENDIF()
    ADD_DEPENDENCIES(benchmarks ${target})
    TARGET_USE_STDTENSOR(${target})
    IF(USE_EXTERN)
        TARGET_LINK_LIBRARIES(${target} stdnn-ops)
    ENDIF()
ENDFUNCTION()

ADD_BENCH(bench-1 tests/bench_pool.cpp)
