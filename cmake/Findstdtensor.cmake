INCLUDE(ExternalProject)

SET(STDTENSOR_GIT_URL https://github.com/lgarithm/stdtensor.git
    CACHE STRING "URL for clone stdtensor")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(libstdtensor
                    GIT_REPOSITORY
                    ${STDTENSOR_GIT_URL}
                    GIT_TAG
                    v0.2.0
                    PREFIX
                    ${PREFIX}
                    CMAKE_ARGS
                    -DCMAKE_INSTALL_PREFIX=${PREFIX}
                    -DBUILD_TESTS=0
                    -DBUILD_EXAMPLES=0
                    -DBUILD_BENCHMARKS=0)

FUNCTION(USE_STDTENSOR target)
    ADD_DEPENDENCIES(${target} libstdtensor)
ENDFUNCTION()
