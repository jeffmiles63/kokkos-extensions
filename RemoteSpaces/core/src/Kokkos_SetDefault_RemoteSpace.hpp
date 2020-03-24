#ifndef __SET_DEFAULT_REMOTE_SPACES
#define __SET_DEFAULT_REMOTE_SPACES

namespace Kokkos {

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
typedef NVSHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_SHMEMSPACE
typedef SHMEMSpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_MPISPACE
typedef MPISpace DefaultRemoteMemorySpace;
#else
#ifdef KOKKOS_ENABLE_QUOSPACE
typedef QUOSpace DefaultRemoteMemorySpace;
#endif
#endif
#endif
#endif

}  // namespace Kokkos

#endif
