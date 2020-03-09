
CMAKE_POLICY(SET CMP0011 NEW)

find_package(PkgConfig REQUIRED)
pkg_search_module(SICM REQUIRED sicm)

KOKKOS_TPL_OPTION(SICM On)

#KOKKOS_LINK_TPL(kokkoscore PUBLIC IMPORTED_NAME sicm SICM)
target_include_directories(kokkoscore PUBLIC "${SICM_INCLUDE_DIRS}")
target_link_libraries(kokkoscore PUBLIC "${SICM_LDFLAGS}")
