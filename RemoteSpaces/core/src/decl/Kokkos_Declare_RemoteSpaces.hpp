#ifndef __KOKKOS_DECL_REMOTESPACES_H
#define __KOKKOS_DECL_REMOTESPACES_H

#include <Kokkos_RemoteSpaces.hpp>

#ifdef KOKKOS_ENABLE_QUOSPACE
#include <Kokkos_QUOSpace.hpp>
#endif

#ifdef KOKKOS_ENABLE_SHMEMSPACE
#include <Kokkos_SHMEMSpace.hpp>
#endif

#ifdef KOKKOS_ENABLE_NVSHMEMSPACE
#include <Kokkos_NVSHMEM_Space.hpp>
#endif

#ifdef KOKKOS_ENABLE_MPISPACE
#include <Kokkos_MPISpace.hpp>
#endif

#endif  // __KOKKOS_DECL_REMOTESPACES_H
