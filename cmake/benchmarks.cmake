INCLUDE(ExternalProject)

SET(GBENCH_GIT_URL
    https://github.com/google/benchmark.git
    CACHE
    STRING
    "URL for clone google benchmark")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(gbench-repo
                    GIT_REPOSITORY
                    ${GBENCH_GIT_URL}
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DCMAKE_BUILD_TYPE=Release
                    -DCMAKE_CXX_FLAGS=-std=c++11
                    -DBENCHMARK_ENABLE_TESTING=0)

LINK_DIRECTORIES(${PREFIX}/lib)

FIND_PACKAGE(Threads REQUIRED)
FUNCTION(ADD_BENCH target)
    ADD_EXECUTABLE(${target} ${ARGN})
    TARGET_LINK_LIBRARIES(${target} benchmark benchmark_main Threads::Threads)
    ADD_DEPENDENCIES(${target} gbench-repo)
ENDFUNCTION()

ADD_BENCH(bench-1 tests/bench_pool.cpp)
