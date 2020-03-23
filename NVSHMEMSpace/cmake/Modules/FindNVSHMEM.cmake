find_package(MPI)

find_path(_nvshmem_root
          NAMES include/shmemx.h
          HINTS $ENV{NVSHMEM_ROOT} $ENV{NVSHMEM_DIR} ${NVSHMEM_ROOT} ${NVSHMEM_DIR}
          )

find_library(_nvshmem_lib
             NAMES libnvshmem.a
             HINTS ${_nvshmem_root}/lib ${_nvshmem_root}/lib64)

find_path(_nvshmem_include_dir
          NAMES shmemx.h
          HINTS ${_nvshmem_root}/include)

if ((NOT ${_nvshmem_root})
        OR (NOT ${_nvshmem_lib})
        OR (NOT ${_nvshmem_include_dir}))
  set(_fail_msg "Could NOT find NVSHMEM (set NVSHMEM_DIR or NVSHMEM_ROOT to point to install)")
elseif ((NOT ${MPI_FOUND}) OR (NOT ${MPI_CXX_FOUND}))
  set(_fail_msg "Could NOT find NVSHMEM (missing MPI)")
else()
  set(_fail_msg "Could NOT find NVSHMEM")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NVSHMEM ${_fail_msg}
                                  _nvshmem_root
                                  _nvshmem_lib
                                  _nvshmem_include_dir
                                  MPI_FOUND
                                  MPI_CXX_FOUND
                                  )

add_library(NVSHMEM::NVSHMEM UNKNOWN IMPORTED)
set_target_properties(NVSHMEM::NVSHMEM PROPERTIES
                      IMPORTED_LOCATION ${_nvshmem_lib}
                      INTERFACE_INCLUDE_DIRECTORIES ${_nvshmem_include_dir}
                      )

set(NVSHMEM_DIR ${_nvshmem_root})

mark_as_advanced(
  _nvshmem_library
  _nvshmem_include_dir
)
