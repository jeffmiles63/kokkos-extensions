#ifndef __KOKKOS_FWD_REMOTESPACES
#define __KOKKOS_FWD_REMOTESPACES

#include <KokkosCore_config_remotespaces.h>

#ifdef KOKKOS_ENABLE_QUOSPACE
class QUOSpace;
#endif

#ifdef KOKKOS_ENABLE_SHMEMSPACE
class SHMEMSpace;
#endif

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
class NVSHMEMSpace;
#endif

#ifdef KOKKOS_ENABLE_MPISPACE
class MPISpace;
#endif

#endif  // __KOKKOS_FWD_REMOTESPACES
