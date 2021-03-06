INCLUDE(ExternalProject)

SET(STDTRACER_GIT_URL
    https://github.com/stdml/stdtracer
    CACHE STRING "URL for clone stdtracer")

SET(STDTRACER_GIT_TAG
    master
    CACHE STRING "git tag for checkout stdtracer")

SET(PREFIX ${CMAKE_SOURCE_DIR}/3rdparty)

EXTERNALPROJECT_ADD(
    libstdtracer
    GIT_REPOSITORY ${STDTRACER_GIT_URL}
    GIT_TAG ${STDTRACER_GIT_TAG}
    PREFIX ${PREFIX}
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${PREFIX} -DBUILD_TESTS=0
               -DBUILD_EXAMPLES=0)

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/3rdparty/src/libstdtracer/include)
