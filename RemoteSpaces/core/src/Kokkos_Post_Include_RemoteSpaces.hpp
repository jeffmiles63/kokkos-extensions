#ifndef __KOKKOS_POST_INCLUDE_REMOTESPACES
#define __KOKKOS_POST_INCLUDE_REMOTESPACES

#include <Kokkos_SetDefault_RemoteSpace.hpp>
#include <string>

namespace Kokkos {

template <typename ViewType, class... Args>
ViewType allocate_symmetric_remote_view(const char* const label, int num_ranks,
                                        int* rank_list, Args... args) {
  typedef typename ViewType::memory_space t_mem_space;
  typedef typename ViewType::array_layout t_layout;

  t_mem_space space;
  int64_t size = ViewType::required_allocation_size(1, args...);
  space.impl_set_allocation_mode(Kokkos::Symmetric);
  space.impl_set_rank_list(rank_list);
  space.impl_set_extent(size);
  t_layout layout(num_ranks, args...);
  return ViewType(Kokkos::view_alloc(std::string(label), space), num_ranks,
                  args...);
}

}  // namespace Kokkos

#include <impl/Kokkos_NVSHMEM_ViewMapping.hpp>

#endif  // __KOKKOS_POST_INCLUDE_REMOTESPACES
