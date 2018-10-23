# package
# https://gitlab.kitware.com/cmake/community/wikis/doc/cpack/PackageGenerators

INCLUDE(CPack)

SET(CPACK_GENERATOR "TGZ")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "lg4869@outlook.com")
IF((${CMAKE_SYSTEM_NAME} MATCHES "Linux"))
    SET(CPACK_GENERATOR "TGZ;DEB") # SET(CPACK_GENERATOR "TGZ;RPM")
ENDIF()
