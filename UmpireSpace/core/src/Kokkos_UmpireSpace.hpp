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

#include "umpire/ResourceManager.hpp"
#include "umpire/Allocator.hpp"
/*--------------------------------------------------------------------------*/

namespace Kokkos {

namespace Impl {

void umpire_to_umpire_deep_copy(void*, const void*, size_t, bool offset = true);
void host_to_umpire_deep_copy(void*, const void*, size_t, bool offset = true);
void umpire_to_host_deep_copy(void*, const void*, size_t, bool offset = true);
void* umpire_allocate(const char*, size_t);
void umpire_deallocate(const char* name, void* const arg_alloc_ptr,
                       const size_t);
umpire::Allocator get_allocator(const char* name);

template <class MemorySpace>
inline const char* umpire_space_name(const MemorySpace& default_device) {
  if (std::is_same<MemorySpace, Kokkos::HostSpace>::value) return "HOST";
#if defined(KOKKOS_ENABLE_CUDA)
  if (std::is_same<MemorySpace, Kokkos::CudaSpace>::value) return "DEVICE";
  if (std::is_same<MemorySpace, Kokkos::CudaUVMSpace>::value) return "UM";
  if (std::is_same<MemorySpace, Kokkos::CudaHostPinnedSpace>::value)
    return "HOSTPINNED";
#endif
}
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
    return Impl::umpire_allocate(m_AllocatorName, arg_alloc_size);
  }

  /**\brief  Deallocate untracked memory in the space */
  inline void deallocate(void* const arg_alloc_ptr,
                         const size_t arg_alloc_size) const {
    return Impl::umpire_deallocate(m_AllocatorName, arg_alloc_ptr,
                                   arg_alloc_size);
  }

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

  static constexpr bool is_host_accessible_space() {
    return Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                           upstream_memory_space>::accessible;
  }

 private:
  using upstream_memory_space = MemorySpace;
  const char* m_AllocatorName;
  static constexpr const char* m_name = "Umpire";
  friend class Kokkos::Impl::SharedAllocationRecord<
      Kokkos::UmpireSpace<upstream_memory_space>, void>;
};

using UmpireHostSpace = UmpireSpace<Kokkos::HostSpace>;
#if defined(KOKKOS_ENABLE_CUDA)
using UmpireCudaSpace           = UmpireSpace<Kokkos::CudaSpace>;
using UmpireCudaUVMSpace        = UmpireSpace<Kokkos::CudaUVMSpace>;
using UmpireCudaHostPinnedSpace = UmpireSpace<Kokkos::CudaHostPinnedSpace>;
#endif

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::UmpireHostSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::UmpireHostSpace, Kokkos::HostSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, Kokkos::UmpireHostSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::UmpireHostSpace, Kokkos::CudaSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::UmpireCudaSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::UmpireCudaSpace, Kokkos::HostSpace> {
  enum { assignable = false };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::CudaSpace, Kokkos::UmpireCudaSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::UmpireCudaSpace, Kokkos::CudaSpace> {
  enum { assignable = true };
  enum { accessible = true };
  enum { deepcopy = true };
};
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
class SharedAllocationRecord<
    MemorySpace,
    typename std::enable_if<is_umpire_space<MemorySpace>::value, void>::type>
    : public SharedAllocationRecord<void, void> {
 private:
  friend MemorySpace;

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

  const MemorySpace m_space;

 protected:
  inline ~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::deallocateData(
          Kokkos::Profiling::SpaceHandle(MemorySpace::name()),
          RecordBase::m_alloc_ptr->m_label, data(), size());
    }
#endif

    m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                       SharedAllocationRecord<void, void>::m_alloc_size);
  }
  SharedAllocationRecord() = default;

  inline SharedAllocationRecord(
      const MemorySpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate)
      : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
            &SharedAllocationRecord<MemorySpace, void>::s_root_record,
#endif
            Kokkos::Impl::checked_allocation_with_header(arg_space, arg_label,
                                                         arg_alloc_size),
            sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
        m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
    if (Kokkos::Profiling::profileLibraryLoaded()) {
      Kokkos::Profiling::allocateData(
          Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
          arg_alloc_size);
    }
#endif

#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    // is_host_accessible_space implies that header is in host space, so we
    // can access it directly
    if (MemorySpace::is_host_accessible_space()) {
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
      Kokkos::Impl::host_to_umpire_deep_copy(RecordBase::m_alloc_ptr, &header,
                                             sizeof(SharedAllocationHeader),
                                             false);
    }
#endif
  }

 public:
  inline std::string get_label() const {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    if (MemorySpace::is_host_accessible_space()) {
      return std::string(RecordBase::head()->m_label);
    } else {
      // we don't know where the umpire pointer lives, so it is best to create a
      // local header, then deep copy from umpire to host and use the local.
      SharedAllocationHeader header;
      Kokkos::Impl::umpire_to_host_deep_copy(
          &header, RecordBase::head(), sizeof(SharedAllocationHeader), false);

      return std::string(header.m_label);
    }
#else
    return "";
#endif
  }

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const MemorySpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    return (SharedAllocationRecord*)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  inline static void* allocate_tracked(const MemorySpace& arg_space,
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

    Kokkos::Impl::DeepCopy<MemorySpace, MemorySpace>(
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
      Kokkos::Impl::umpire_to_host_deep_copy(
          &head, head_dev, sizeof(SharedAllocationHeader), false);
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
  inline static void print_records(std::ostream&, const MemorySpace& s,
                                   bool detail = false) {
    SharedAllocationRecord<void, void>::print_host_accessible_records(
        s, "UmpireSpace", &s_root_record, detail);
  }
#else
  inline static void print_records(std::ostream&, const MemorySpace&, bool) {
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

template <class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireHostSpace, Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    host_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    host_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::UmpireHostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    umpire_to_host_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    umpire_to_host_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireHostSpace, Kokkos::UmpireHostSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    umpire_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    umpire_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireCudaSpace, Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    host_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    host_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::UmpireCudaSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    umpire_to_host_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    umpire_to_host_deep_copy(dst, src, n);
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireCudaSpace, Kokkos::UmpireCudaSpace,
                ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    umpire_to_umpire_deep_copy(dst, src, n);
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    umpire_to_umpire_deep_copy(dst, src, n);
    exec.fence();
  }
};
}  // namespace Impl

}  // namespace Kokkos

#endif  // #define KOKKOS_UMPIRESPACE_HPP
