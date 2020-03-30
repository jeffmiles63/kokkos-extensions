
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../../cmake/Modules")

IF( KOKKOS_ENABLE_QUOMEMSPACE)

   KOKKOS_ADD_EXECUTABLE_AND_TEST(
      Test_QUO_OpenMP
      SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/Test_Main.cpp
      ${CMAKE_CURRENT_LIST_DIR}/Test_Allocation.cpp)

   target_compile_definitions(KokkosCore_Test_QUO_OpenMP PUBLIC KOKKOS_ENABLE_QUO_TEST)
ENDIF()

IF( KOKKOS_ENABLE_SHMEMSPACE)

   KOKKOS_ADD_EXECUTABLE_AND_TEST(
      Test_SHMEM_OpenMP
      SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/Test_Main.cpp
      ${CMAKE_CURRENT_LIST_DIR}/Test_Allocation.cpp)

   target_compile_definitions(KokkosCore_Test_SHMEM_OpenMP PUBLIC KOKKOS_ENABLE_SHMEM_TEST)
ENDIF()

IF( KOKKOS_ENABLE_MPISPACE)

   KOKKOS_ADD_TEST_EXECUTABLE(
      KokkosCore_Test_MPI_OpenMP
      SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/Test_Main.cpp
      ${CMAKE_CURRENT_LIST_DIR}/Test_Allocation.cpp)

   target_compile_definitions(KokkosCore_Test_MPI_OpenMP PUBLIC KOKKOS_ENABLE_MPI_TEST)

   KOKKOS_ADD_TEST( NAME KokkosCore_Test_MPI_OpenMP
                    EXE  mpirun
                    FAIL_REGULAR_EXPRESSION "FAILED"
                    CMD_ARGS -n 1 KokkosCore_Test_MPI_OpenMP
                  )
ENDIF()

IF( KOKKOS_ENABLE_NVSHMEMSPACE)

   KOKKOS_ADD_TEST_EXECUTABLE(
      KokkosCore_Test_NVSHMEM_Cuda
      SOURCES
      ${CMAKE_CURRENT_LIST_DIR}/Test_Main.cpp
      ${CMAKE_CURRENT_LIST_DIR}/Test_Allocation.cpp)

   target_compile_definitions(KokkosCore_Test_NVSHMEM_Cuda PUBLIC KOKKOS_ENABLE_NVSHMEM_TEST)

   KOKKOS_ADD_TEST( NAME KokkosCore_Test_NVSHMEM_Cuda
                    EXE  mpirun
                    FAIL_REGULAR_EXPRESSION "FAILED"
                    CMD_ARGS -n 1 KokkosCore_Test_NVSHMEM_Cuda
                  )
ENDIF()
