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

#include <cstdio>
#include <algorithm>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_MemorySpace.hpp>
#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif

/*--------------------------------------------------------------------------*/

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <sstream>
#include <cstring>

#include <Kokkos_UmpireSpace.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Kokkos_Atomic.hpp>

#include "umpire/op/MemoryOperationRegistry.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

/* umpire_to_umpire_deep_copy: copy from umpire ptr to umpire ptr
 *                             umpire allocation records for each space
 *                             are used directly (accessed from resource
 *                             manager via the findAllocationRecord method
 *  Note: This function is called from the DeepCopy<> templates but also
 *        from the Kokkos::Impl::SharedAllocationRecord.  if called from the
 *        templates, offset = true which means that the pointer is offset
 *        by the size of the SharedAllocationHeader.  if called from
 *        SharedAllocationRecord, the pointer is the same as that provided
 *        from Umpire.
 */
void umpire_to_umpire_deep_copy(void *dst, const void *src, size_t size,
                                bool offset) {
  auto &rm          = umpire::ResourceManager::getInstance();
  auto &op_registry = umpire::op::MemoryOperationRegistry::getInstance();

  Kokkos::Impl::SharedAllocationHeader *dst_header =
      (Kokkos::Impl::SharedAllocationHeader *)dst;

  Kokkos::Impl::SharedAllocationHeader *src_header =
      (Kokkos::Impl::SharedAllocationHeader *)src;

  if (offset) {
    src_header -= 1;
    dst_header -= 1;
  }

  auto src_alloc_record = rm.findAllocationRecord(src_header);
  std::ptrdiff_t src_offset =
      reinterpret_cast<char *>(src_header) -
              reinterpret_cast<char *>(src_alloc_record->ptr) + offset
          ? sizeof(Kokkos::Impl::SharedAllocationHeader)
          : 0;

  std::size_t src_size = src_alloc_record->size - src_offset;

  auto dst_alloc_record = rm.findAllocationRecord(dst_header);
  std::ptrdiff_t dst_offset =
      reinterpret_cast<char *>(dst_header) -
              reinterpret_cast<char *>(dst_alloc_record->ptr) + offset
          ? sizeof(Kokkos::Impl::SharedAllocationHeader)
          : 0;

  std::size_t dst_size = dst_alloc_record->size - dst_offset;

  UMPIRE_REPLAY(R"( "event": "copy", "payload": { "src": ")"
                << src_header << R"(", src_offset: ")" << src_offset
                << R"(", "dest": ")" << dst_header << R"(", dst_offset: ")"
                << dst_offset << R"(",  "size": )" << size
                << R"(, "src_allocator_ref": ")" << src_alloc_record->strategy
                << R"(", "dst_allocator_ref": ")" << dst_alloc_record->strategy
                << R"(" } )");

  if (size > src_size) {
    UMPIRE_ERROR("Copy asks for more that resides in source copy: "
                 << size << " -> " << src_size);
  }

  if (size > dst_size) {
    UMPIRE_ERROR("Not enough resource in destination for copy: "
                 << size << " -> " << dst_size);
  }

  auto op = op_registry.find("COPY", src_alloc_record->strategy,
                             dst_alloc_record->strategy);

  op->transform(const_cast<void *>(src), &dst,
                const_cast<umpire::util::AllocationRecord *>(src_alloc_record),
                const_cast<umpire::util::AllocationRecord *>(dst_alloc_record),
                size);
}

/* host_to_umpire_deep_copy - copy from kokkos host allocated ptr to umpire
 * allocated ptr. same rules apply as above, but only for the dst pointer.
 *
 */
void host_to_umpire_deep_copy(void *dst, const void *src, size_t size,
                              bool offset) {
  auto &rm           = umpire::ResourceManager::getInstance();
  auto &op_registry  = umpire::op::MemoryOperationRegistry::getInstance();
  auto hostAllocator = rm.getAllocator("HOST");

  Kokkos::Impl::SharedAllocationHeader *dst_header =
      (Kokkos::Impl::SharedAllocationHeader *)dst;
  if (offset) dst_header -= 1;

  auto dst_alloc_record = rm.findAllocationRecord(dst_header);
  std::ptrdiff_t dst_offset =
      reinterpret_cast<char *>(dst_header) -
              reinterpret_cast<char *>(dst_alloc_record->ptr) + offset
          ? sizeof(Kokkos::Impl::SharedAllocationHeader)
          : 0;

  std::size_t dst_size = dst_alloc_record->size - dst_offset;

  if (size > dst_size) {
    UMPIRE_ERROR("Copy asks for more that will fit in the destination: "
                 << size << " -> " << dst_size);
  }

  // Have to create a "fake" host allocator strategy to get the correct
  // Operation object
  umpire::util::AllocationRecord src_alloc_record{
      nullptr, size, hostAllocator.getAllocationStrategy()};

  auto op = op_registry.find("COPY", src_alloc_record.strategy,
                             dst_alloc_record->strategy);

  op->transform(const_cast<void *>(src), &dst,
                const_cast<umpire::util::AllocationRecord *>(&src_alloc_record),
                const_cast<umpire::util::AllocationRecord *>(dst_alloc_record),
                size);
}

/* umpire_to_host_deep_copy - copy from umpire allocated ptr to kokkos host
 * allocated ptr. same rules apply as above, but only for the src pointer.
 *
 */
void umpire_to_host_deep_copy(void *dst, const void *src, size_t size,
                              bool offset) {
  auto &rm           = umpire::ResourceManager::getInstance();
  auto &op_registry  = umpire::op::MemoryOperationRegistry::getInstance();
  auto hostAllocator = rm.getAllocator("HOST");

  Kokkos::Impl::SharedAllocationHeader *src_header =
      (Kokkos::Impl::SharedAllocationHeader *)src;
  if (offset) src_header -= 1;

  auto src_alloc_record = rm.findAllocationRecord(src_header);
  std::ptrdiff_t src_offset =
      reinterpret_cast<char *>(src_header) -
              reinterpret_cast<char *>(src_alloc_record->ptr) + offset
          ? sizeof(Kokkos::Impl::SharedAllocationHeader)
          : 0;

  std::size_t src_size = src_alloc_record->size - src_offset;

  if (size > src_size) {
    UMPIRE_ERROR("Copy asks for more that resides in source copy: "
                 << size << " -> " << src_size);
  }

  // Have to create a "fake" host allocator strategy to get the correct
  // Operation object
  umpire::util::AllocationRecord dst_alloc_record{
      nullptr, size, hostAllocator.getAllocationStrategy()};

  auto op = op_registry.find("COPY", src_alloc_record->strategy,
                             dst_alloc_record.strategy);

  op->transform(const_cast<void *>(src), &dst,
                const_cast<umpire::util::AllocationRecord *>(src_alloc_record),
                const_cast<umpire::util::AllocationRecord *>(&dst_alloc_record),
                size);
}

umpire::Allocator get_allocator(const char *name) {
  auto &rm = umpire::ResourceManager::getInstance();
  return rm.getAllocator(name);
}

void *umpire_allocate(const char *name, const size_t arg_alloc_size) {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  constexpr uintptr_t alignment = Kokkos::Impl::MEMORY_ALIGNMENT;

  void *ptr = nullptr;

  if (arg_alloc_size) {
    // Over-allocate to and round up to guarantee proper alignment.
    size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

    auto allocator = get_allocator(name);
    ptr            = allocator.allocate(size_padded);
  }

  if (ptr == nullptr) {
    Experimental::RawMemoryAllocationFailure::FailureMode failure_mode =
        Experimental::RawMemoryAllocationFailure::FailureMode::OutOfMemoryError;

    throw Kokkos::Experimental::RawMemoryAllocationFailure(
        arg_alloc_size, alignment, failure_mode,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::
            StdMalloc);
  }

  return ptr;
}

void umpire_deallocate(const char *name, void *const arg_alloc_ptr,
                       const size_t) {
  if (arg_alloc_ptr) {
    auto allocator = get_allocator(name);
    allocator.deallocate(const_cast<void *>(arg_alloc_ptr));
  }
}

}  // namespace Impl
}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------
