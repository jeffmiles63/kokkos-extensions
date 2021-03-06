/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_UMPIRESPACE_HPP
#define KOKKOS_UMPIRESPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_HostSpace.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>
#include <impl/Kokkos_MemorySpace.hpp>

#include <impl/Kokkos_Tools.hpp>

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

bool test_umpire_from_ptr(const void* ptr, bool offset);
void umpire_to_umpire_deep_copy(void*, const void*, size_t, bool offset = true);
void kokkos_to_umpire_deep_copy(const char*, void*, const void*, size_t,
                                bool offset = true);
void umpire_to_kokkos_deep_copy(const char*, void*, const void*, size_t,
                                bool offset = true);
void* umpire_allocate(const char*, size_t);
void umpire_deallocate(const char* name, void* const arg_alloc_ptr,
                       const size_t);
umpire::Allocator get_allocator(const char* name);

template <class MemorySpace>
inline constexpr
    typename std::enable_if<std::is_same<MemorySpace, Kokkos::HostSpace>::value,
                            const char*>::type
    umpire_space_name(const MemorySpace& default_device) {
  return "HOST";
}
#if defined(KOKKOS_ENABLE_CUDA)
template <class MemorySpace>
inline constexpr
    typename std::enable_if<std::is_same<MemorySpace, Kokkos::CudaSpace>::value,
                            const char*>::type
    umpire_space_name(const MemorySpace& default_device) {
  return "DEVICE";
}
template <class MemorySpace>
inline constexpr typename std::enable_if<
    std::is_same<MemorySpace, Kokkos::CudaUVMSpace>::value, const char*>::type
umpire_space_name(const MemorySpace& default_device) {
  return "UM";
}
template <class MemorySpace>
inline constexpr typename std::enable_if<
    std::is_same<MemorySpace, Kokkos::CudaHostPinnedSpace>::value,
    const char*>::type
umpire_space_name(const MemorySpace& default_device) {
  return "PINNED";
}
#endif

}  // namespace Impl

/// \class UmpireSpace
/// \brief Memory management for host memory.
///
/// UmpireSpace is a memory space that governs host memory.  "Host"
/// memory means the usual CPU-accessible memory.
template <class MemorySpace>
class UmpireSpace {
 public:
  //! Tag this class as a kokkos memory space
  using memory_space    = UmpireSpace;
  using umpire_space    = UmpireSpace;
  using size_type       = size_t;
  using execution_space = typename MemorySpace::execution_space;

  //! This memory space preferred device_type
  typedef Kokkos::Device<execution_space, memory_space> device_type;

  /**\brief  Default memory space instance */
  explicit UmpireSpace(const char* name_) : m_AllocatorName(name_) {
    // somehow need to check that the name is consistent with the upstream
    // memory space
  }

  /* Default allocation mechanism, assume the Umpire allocator is HOST */
  UmpireSpace()
      : m_AllocatorName(Impl::umpire_space_name(upstream_memory_space())) {}

  UmpireSpace(UmpireSpace&& rhs)      = default;
  UmpireSpace(const UmpireSpace& rhs) = default;
  UmpireSpace& operator=(UmpireSpace&&) = default;
  UmpireSpace& operator=(const UmpireSpace&) = default;
  ~UmpireSpace()                             = default;

  /**\brief  Allocate untracked memory in the space */
  inline void* allocate(const size_t arg_alloc_size) const {
     return allocate("[unlabelled]", arg_alloc_size);
  }

  /**\brief  Allocate untracked memory in the space */
  inline void* allocate(const char* arg_label, 
                        const size_t arg_alloc_size,
                        const size_t arg_logical_size = 0) const {
    const size_t reported_size =
        (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
    static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");
   
    void * ptr = Impl::umpire_allocate(m_AllocatorName, arg_alloc_size);
    if (ptr != nullptr && Kokkos::Profiling::profileLibraryLoaded()) {
         Kokkos::Profiling::allocateData(
             Kokkos::Profiling::make_space_handle(name()), arg_label, ptr,
             reported_size);
    }
    return ptr;
  }

  /**\brief  Deallocate untracked memory in the space */
  inline void deallocate(void* const arg_alloc_ptr,
                         const size_t arg_alloc_size) const {
      return deallocate( "[unlabelled]",  arg_alloc_ptr, arg_alloc_size );
  }

  inline void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const {
    if (arg_alloc_ptr != nullptr && arg_alloc_size > 0) {
      size_t reported_size =
          (arg_logical_size > 0) ? arg_logical_size : arg_alloc_size;
      if (Kokkos::Profiling::profileLibraryLoaded()) {
        Kokkos::Profiling::deallocateData(
            Kokkos::Profiling::make_space_handle(name()), arg_label,
            arg_alloc_ptr, reported_size);
      }
      Impl::umpire_deallocate(m_AllocatorName, arg_alloc_ptr, arg_alloc_size);
    }
  }

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  static constexpr bool is_host_accessible_space() {
    return Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                           upstream_memory_space>::accessible;
  }

  template <class StrategyType, class... Args>
  static inline void make_new_allocator(const char* name_ext, Args... args) {
    std::string space_name = Impl::umpire_space_name(upstream_memory_space());
    std::string new_alloc_name = space_name + name_ext;
    auto& rm                   = umpire::ResourceManager::getInstance();
    auto pooled_allocator      = rm.makeAllocator<StrategyType>(
        new_alloc_name, Impl::get_allocator(space_name.c_str()), args...);
  }

 private:
  using upstream_memory_space = MemorySpace;
  const char* m_AllocatorName;
  static constexpr const char* m_name = "Umpire";
  friend class Kokkos::Impl::SharedAllocationRecord<
      Kokkos::UmpireSpace<upstream_memory_space>, void>;
};

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::HostSpace,
                         Kokkos::UmpireSpace<InternalMemorySpace>> {
  using internal_memory_space_access =
      MemorySpaceAccess<Kokkos::HostSpace, InternalMemorySpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::UmpireSpace<InternalMemorySpace>,
                         Kokkos::HostSpace> {
  using internal_memory_space_access =
      MemorySpaceAccess<InternalMemorySpace, Kokkos::HostSpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};

#ifdef KOKKOS_ENABLE_CUDA

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::CudaHostPinnedSpace,
                         Kokkos::UmpireSpace<InternalMemorySpace>> {
  using internal_memory_space_access =
      MemorySpaceAccess<Kokkos::CudaHostPinnedSpace, InternalMemorySpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::UmpireSpace<InternalMemorySpace>,
                         Kokkos::CudaHostPinnedSpace> {
  using internal_memory_space_access =
      MemorySpaceAccess<InternalMemorySpace, Kokkos::CudaHostPinnedSpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::CudaUVMSpace,
                         Kokkos::UmpireSpace<InternalMemorySpace>> {
  using internal_memory_space_access =
      MemorySpaceAccess<Kokkos::CudaUVMSpace, InternalMemorySpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::UmpireSpace<InternalMemorySpace>,
                         Kokkos::CudaUVMSpace> {
  using internal_memory_space_access =
      MemorySpaceAccess<InternalMemorySpace, Kokkos::CudaUVMSpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::CudaSpace,
                         Kokkos::UmpireSpace<InternalMemorySpace>> {
  using internal_memory_space_access =
      MemorySpaceAccess<Kokkos::CudaSpace, InternalMemorySpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};

template <class InternalMemorySpace>
struct MemorySpaceAccess<Kokkos::UmpireSpace<InternalMemorySpace>,
                         Kokkos::CudaSpace> {
  using internal_memory_space_access =
      MemorySpaceAccess<InternalMemorySpace, Kokkos::CudaSpace>;
  enum { assignable = internal_memory_space_access::assignable };
  enum { accessible = internal_memory_space_access::accessible };
  enum { deepcopy = internal_memory_space_access::deepcopy };
};
#endif

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class T, class Enable = void>
struct is_umpire_space {
  enum { value = 0 };
};

template <class T>
struct is_umpire_space<
    T, typename std::enable_if<std::is_same<
           typename std::remove_cv<T>::type,
           typename std::remove_cv<typename T::umpire_space>::type>::value>::
           type> {
  enum { value = 1 };
};

template <class MemorySpace>
class SharedAllocationRecord<Kokkos::UmpireSpace<MemorySpace>, void>
    : public SharedAllocationRecord<void, void> {
 private:
  using memory_space = Kokkos::UmpireSpace<MemorySpace>;
  friend memory_space;

  using RecordBase = SharedAllocationRecord<void, void>;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  inline static void deallocate(RecordBase* arg_rec) {
    delete static_cast<SharedAllocationRecord*>(arg_rec);
  }

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this UmpireSpace instance
   */
  inline static RecordBase s_root_record;
#endif

  const memory_space m_space;

 protected:
  inline ~SharedAllocationRecord() {
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(
          Kokkos::Profiling::make_space_handle(memory_space::name()),
          RecordBase::m_alloc_ptr->m_label, data(), size());
    }
    m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                       SharedAllocationRecord<void, void>::m_alloc_size);
  }
  SharedAllocationRecord() = default;

  inline SharedAllocationRecord(
      const memory_space& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
      : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
            &SharedAllocationRecord<memory_space, void>::s_root_record,
#endif
            Kokkos::Impl::checked_allocation_with_header(arg_space, arg_label,
                                                         arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
        m_space(arg_space) {
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::allocateData(
          Kokkos::Profiling::make_space_handle(arg_space.name()), arg_label,
          data(), arg_alloc_size);
    }

#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    // is_host_accessible_space implies that header is in host space, so we
    // can access it directly
    if (memory_space::is_host_accessible_space()) {
      // Fill in the Header information
      RecordBase::m_alloc_ptr->m_record =
          static_cast<SharedAllocationRecord<void, void>*>(this);

      strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
              SharedAllocationHeader::maximum_label_length);
      // Set last element zero, in case c_str is too long
      RecordBase::m_alloc_ptr
          ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
    } else {
      SharedAllocationHeader header;

      // Fill in the Header information
      header.m_record = static_cast<SharedAllocationRecord<void, void>*>(this);

      strncpy(header.m_label, arg_label.c_str(),
              SharedAllocationHeader::maximum_label_length);
      // Set last element zero, in case c_str is too long
      header.m_label[SharedAllocationHeader::maximum_label_length - 1] =
          (char)0;

      // Copy to device memory
      Kokkos::Impl::kokkos_to_umpire_deep_copy(
          "HOST", RecordBase::m_alloc_ptr, &header,
          sizeof(SharedAllocationHeader), false);
    }
#endif
  }

 public:
  inline std::string get_label() const {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    if (memory_space::is_host_accessible_space()) {
      return std::string(RecordBase::head()->m_label);
    } else {
      // we don't know where the umpire pointer lives, so it is best to create a
      // local header, then deep copy from umpire to host and use the local.
      SharedAllocationHeader header;
      Kokkos::Impl::umpire_to_kokkos_deep_copy(
          "HOST", &header, RecordBase::head(), sizeof(SharedAllocationHeader),
          false);

      return std::string(header.m_label);
    }
#else
    return "";
#endif
  }

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const memory_space& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    return (SharedAllocationRecord*)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  inline static void* allocate_tracked(const memory_space& arg_space,
                                       const std::string& arg_alloc_label,
                                       const size_t arg_alloc_size) {
    if (!arg_alloc_size) return (void*)nullptr;

    SharedAllocationRecord* const r =
        allocate(arg_space, arg_alloc_label, arg_alloc_size);

    RecordBase::increment(r);

    return r->data();
  }

  /**\brief  Reallocate tracked memory in the space */
  inline static void* reallocate_tracked(void* const arg_alloc_ptr,
                                         const size_t arg_alloc_size) {
    SharedAllocationRecord* const r_old = get_record(arg_alloc_ptr);
    SharedAllocationRecord* const r_new =
        allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

    Kokkos::Impl::DeepCopy<memory_space, memory_space>(
        r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

    RecordBase::increment(r_new);
    RecordBase::decrement(r_old);

    return r_new->data();
  }

  /**\brief  Deallocate tracked memory in the space */
  inline static void deallocate_tracked(void* const arg_alloc_ptr) {
    if (arg_alloc_ptr != 0) {
      SharedAllocationRecord* const r = get_record(arg_alloc_ptr);

      RecordBase::decrement(r);
    }
  }

  inline static SharedAllocationRecord* get_record(void* arg_alloc_ptr) {
    using Header       = SharedAllocationHeader;
    using RecordUmpire = SharedAllocationRecord;

    // Copy the header from the allocation
    // cannot determine if it is host or device statically, so we will always
    // deep copy it....

#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    Header head;

    Header const* const head_dev =
        arg_alloc_ptr ? Header::get_header(arg_alloc_ptr) : (Header*)0;

    if (arg_alloc_ptr) {
      Kokkos::Impl::umpire_to_kokkos_deep_copy(
          "HOST", &head, head_dev, sizeof(SharedAllocationHeader), false);
    }

    RecordUmpire* const record = arg_alloc_ptr
                                     ? static_cast<RecordUmpire*>(head.m_record)
                                     : (RecordUmpire*)0;

    if (!arg_alloc_ptr || record->m_alloc_ptr != head_dev) {
      Kokkos::Impl::throw_runtime_exception(std::string(
          "Kokkos::Impl::SharedAllocationRecord< Kokkos::UmpireSpace , "
          "void >::get_record ERROR"));
    }
    return record;
#else
    return (SharedAllocationRecord*)0;
#endif
  }

#ifdef KOKKOS_DEBUG
  inline static void print_records(std::ostream&, const memory_space& s,
                                   bool detail = false) {
    SharedAllocationRecord<void, void>::print_host_accessible_records(
        s, "UmpireSpace", &s_root_record, detail);
  }
#else
  inline static void print_records(std::ostream&, const memory_space&, bool) {
    throw_runtime_exception(
        "SharedAllocationRecord<UmpireSpace>::print_records only works with "
        "KOKKOS_DEBUG enabled");
  }
#endif
};

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireSpace<InternalMemorySpace>, Kokkos::HostSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }
  void internal_copy(void* dst, const void* src, size_t n) {
    bool dst_umpire = test_umpire_from_ptr(dst, true);
    if (dst_umpire) {
      kokkos_to_umpire_deep_copy("HOST", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<InternalMemorySpace, Kokkos::HostSpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::UmpireSpace<InternalMemorySpace>,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }
  void internal_copy(void* dst, const void* src, size_t n) {
    bool src_umpire = test_umpire_from_ptr(src, true);
    if (src_umpire) {
      umpire_to_kokkos_deep_copy("HOST", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<Kokkos::HostSpace, InternalMemorySpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

#if defined(KOKKOS_ENABLE_CUDA)
template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireSpace<InternalMemorySpace>, Kokkos::CudaSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }
  void internal_copy(void* dst, const void* src, size_t n) {
    bool dst_umpire = test_umpire_from_ptr(dst, true);
    if (dst_umpire) {
      kokkos_to_umpire_deep_copy("DEVICE", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<InternalMemorySpace, Kokkos::CudaSpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::CudaSpace, Kokkos::UmpireSpace<InternalMemorySpace>,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }
  void internal_copy(void* dst, const void* src, size_t n) {
    bool src_umpire = test_umpire_from_ptr(src, true);
    if (src_umpire) {
      umpire_to_kokkos_deep_copy("DEVICE", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<Kokkos::CudaSpace, InternalMemorySpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireSpace<InternalMemorySpace>,
                Kokkos::CudaHostPinnedSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }
  void internal_copy(void* dst, const void* src, size_t n) {
    bool dst_umpire = test_umpire_from_ptr(dst, true);
    if (dst_umpire) {
      kokkos_to_umpire_deep_copy("PINNED", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<InternalMemorySpace, Kokkos::CudaHostPinnedSpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::CudaHostPinnedSpace,
                Kokkos::UmpireSpace<InternalMemorySpace>, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }
  void internal_copy(void* dst, const void* src, size_t n) {
    bool src_umpire = test_umpire_from_ptr(src, true);
    if (src_umpire) {
      umpire_to_kokkos_deep_copy("PINNED", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<Kokkos::CudaHostPinnedSpace, InternalMemorySpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireSpace<InternalMemorySpace>, Kokkos::CudaUVMSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }
  void internal_copy(void* dst, const void* src, size_t n) {
    bool dst_umpire = test_umpire_from_ptr(dst, true);
    if (dst_umpire) {
      kokkos_to_umpire_deep_copy("UM", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<InternalMemorySpace, Kokkos::CudaUVMSpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

template <class InternalMemorySpace, class ExecutionSpace>
struct DeepCopy<Kokkos::CudaUVMSpace, Kokkos::UmpireSpace<InternalMemorySpace>,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }

  void internal_copy(void* dst, const void* src, size_t n) {
    bool src_umpire = test_umpire_from_ptr(src, true);
    if (src_umpire) {
      umpire_to_kokkos_deep_copy("UM", dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<Kokkos::CudaUVMSpace, InternalMemorySpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};
#endif  // KOKKOS_ENABLE_CUDA

template <class DestInternalMemorySpace, class SrcInternalMemorySpace,
          class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireSpace<DestInternalMemorySpace>,
                Kokkos::UmpireSpace<SrcInternalMemorySpace>, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) { internal_copy(dst, src, n); }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    internal_copy(dst, src, n);
    exec.fence();
  }

  void internal_copy(void* dst, const void* src, size_t n) {
    bool dst_umpire = test_umpire_from_ptr(dst, true);
    bool src_umpire = test_umpire_from_ptr(src, true);
    if (dst_umpire && src_umpire) {
      umpire_to_umpire_deep_copy(dst, src, n);
    } else if (dst_umpire) {
      kokkos_to_umpire_deep_copy(umpire_space_name(SrcInternalMemorySpace()),
                                 dst, src, n);
    } else if (src_umpire) {
      umpire_to_kokkos_deep_copy(umpire_space_name(DestInternalMemorySpace()),
                                 dst, src, n);
    } else {
      Kokkos::Impl::DeepCopy<DestInternalMemorySpace, SrcInternalMemorySpace,
                             ExecutionSpace>(dst, src, n);
    }
  }
};

}  // namespace Impl

}  // namespace Kokkos

#endif  // #define KOKKOS_UMPIRESPACE_HPP
