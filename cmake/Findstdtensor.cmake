INCLUDE(ExternalProject)

SET(STDTENSOR_GIT_URL
    https://github.com/stdml/stdtensor.git
    CACHE STRING "URL for clone stdtensor")

SET(STDTENSOR_GIT_TAG
    "master"
    CACHE STRING "git tag for checkout stdtensor")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(
    stdtensor-repo
    LOG_DOWNLOAD ON
    LOG_INSTALL ON
    LOG_CONFIGURE ON
    GIT_REPOSITORY ${STDTENSOR_GIT_URL}
    GIT_TAG ${STDTENSOR_GIT_TAG}
    PREFIX ${PREFIX}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_TESTS=0
               -DBUILD_EXAMPLES=0 -DBUILD_BENCHMARKS=0)

FUNCTION(TARGET_USE_STDTENSOR target)
    ADD_DEPENDENCIES(${target} stdtensor-repo)
    TARGET_INCLUDE_DIRECTORIES(${target}
                               PRIVATE ${CMAKE_SOURCE_DIR}/3rdparty/include)
ENDFUNCTION()
