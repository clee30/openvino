// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/x64/subgraph.hpp"

#include <csignal>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <numeric>
#include <oneapi/dnnl/dnnl_common.hpp>
#include <string>
#include <utility>
#include <vector>

#include "cache/multi_cache.h"
#include "cpu_types.h"
#include "dnnl_extension_utils.h"
#include "emitters/snippets/cpu_runtime_configurator.hpp"
#include "emitters/snippets/input_repacker.hpp"
#include "emitters/snippets/jit_snippets_call_args.hpp"
#include "emitters/snippets/x64/cpu_generator.hpp"
#include "emitters/snippets/x64/kernel_executors/brgemm_copy_b.hpp"
#include "memory_desc/blocked_memory_desc.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "nodes/executors/subgraph.hpp"
#include "openvino/core/except.hpp"
#include "openvino/core/parallel.hpp"
#include "utils/general_utils.h"

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
#    include "emitters/snippets/x64/jit_segfault_detector_emitter.hpp"
#endif

namespace ov::intel_cpu {

namespace {
inline void parallel4d_repacking(const BrgemmCopyBKernel* ker,
                                 const VectorDims& dom,
                                 const VectorDims& in_str,
                                 const VectorDims& out_str,
                                 const uint8_t* src,
                                 uint8_t* dst) {
    parallel_for4d(dom[0], dom[1], dom[2], dom[3], [&](size_t d0, size_t d1, size_t d2, size_t d3) {
        BrgemmCopyBKernel::call_args args;
        args.src = src + d0 * in_str[0] + d1 * in_str[1] + d2 * in_str[2] + d3 * in_str[3];
        args.tr_src = dst + d0 * out_str[0] + d1 * out_str[1] + d2 * out_str[2] + d3 * out_str[3];
        (*ker)(&args);
    });
};
inline void parallelNd_repacking(const BrgemmCopyBKernel* ker,
                                 const VectorDims& dom,
                                 const VectorDims& in_str,
                                 const VectorDims& out_str,
                                 const uint8_t* src,
                                 uint8_t* dst) {
    const size_t batch = std::accumulate(dom.rbegin() + 2, dom.rend(), 1LU, std::multiplies<>());
    parallel_nt_static(0, [&](const int ithr, const int nthr) {
        BrgemmCopyBKernel::call_args args;
        size_t start = 0;
        size_t end = 0;
        splitter(batch, nthr, ithr, start, end);
        for (size_t iwork = start; iwork < end; ++iwork) {
            const uint8_t* src_u8 = src;
            uint8_t* dst_u8 = dst;
            size_t tmp = iwork;
            for (ptrdiff_t j = static_cast<ptrdiff_t>(dom.size()) - 3; j >= 0; j--) {
                auto idx = tmp % dom[j];
                tmp /= dom[j];

                src_u8 += idx * in_str[j];
                dst_u8 += idx * out_str[j];
            }
            args.src = src_u8;
            args.tr_src = dst_u8;
            (*ker)(&args);
        }
    });
};
}  // namespace

SubgraphExecutor::SubgraphExecutor(const std::shared_ptr<CPURuntimeConfig>& snippet_config,
                                   const std::shared_ptr<SubgraphAttrs>& snippet_attrs,
                                   const std::shared_ptr<SubgraphCodeGenerator>& snippet,
                                   const std::vector<ptrdiff_t>& start_offset_in,
                                   const std::vector<ptrdiff_t>& start_offset_out,
                                   const BufferScratchpadAllocator& allocator,
                                   const ov::intel_cpu::MultiCacheWeakPtr& kernel_cache)
    : SubgraphBaseExecutor(snippet_config,
                           snippet_attrs,
                           snippet,
                           start_offset_in,
                           start_offset_out,
                           allocator,
                           kernel_cache),
      m_input_repackers(snippet_config->input_repackers),
      m_repacking_impl_type(snippet_config->repacking_impl_type) {
    auto external_buffer_size =
        std::accumulate(m_input_repackers.begin(),
                        m_input_repackers.end(),
                        static_cast<size_t>(0),
                        [](size_t sum, const std::pair<size_t, InputRepacker>& p) {
                            auto curr_mem_size = p.second.desc()->getCurrentMemSize();
                            OPENVINO_ASSERT(curr_mem_size != ov::intel_cpu::MemoryDesc::UNDEFINED_SIZE,
                                            "Current repacking buffer memory size is undefined");
                            return sum + curr_mem_size;
                        });

    if (get_repacking_impl_type() == RepackingImplType::IN_PARALLEL) {
        // When external repacking is applied in parallel section,
        // each thread should have own buffer to store repacked data
        external_buffer_size *= m_nthreads;

        // To avoid extra overheads in runtime on vector creation,
        // we initialize `repacked_offsets_by_threads` by default here
        m_repacked_offsets_by_threads.resize(m_nthreads);
        for (size_t i = 0; i < m_repacked_offsets_by_threads.size(); ++i) {
            clean_repacked_offsets(i);
        }

        if (m_tensor_rank == rank6D) {
            init_offset = [](const std::vector<size_t>& offsets, const std::vector<size_t>& indexes, size_t& offset) {
                offset += offsets[0] * indexes[0] + offsets[1] * indexes[1] + offsets[2] * indexes[2] +
                          offsets[3] * indexes[3];
            };
        } else {
            init_offset = [](const std::vector<size_t>& offsets, const std::vector<size_t>& indexes, size_t& offset) {
                for (size_t j = 0; j < indexes.size(); j++) {
                    offset += offsets[j] * indexes[j];
                }
            };
        }
    }

    m_buffer_scratchpad = allocator(m_internal_buffer_size + external_buffer_size);

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    const auto target = std::dynamic_pointer_cast<const CPUTargetMachine>(
        snippet_attrs->snippet->get_generator()->get_target_machine());
    enabled_segfault_detector = target && target->debug_config.enable_segfault_detector;
#endif
}

void SubgraphExecutor::separately_repack_input(const MemoryPtr& src_mem_ptr,
                                               const MemoryPtr& dst_mem_ptr,
                                               const ov::intel_cpu::InputRepacker& input_repacker,
                                               size_t tensor_rank) {
    auto get_offset = [](const BlockedMemoryDescPtr& desc) {
        return static_cast<ptrdiff_t>(desc->getOffsetPadding() * desc->getPrecision().size());
    };

    const auto* src_ptr =
        src_mem_ptr->getDataAs<const uint8_t>() + get_offset(src_mem_ptr->getDescWithType<BlockedMemoryDesc>());
    auto* dst_ptr = dst_mem_ptr->getDataAs<uint8_t>() + get_offset(dst_mem_ptr->getDescWithType<BlockedMemoryDesc>());

    VectorDims dom;
    const auto& shape = dst_mem_ptr->getShape().getDims();
    OPENVINO_ASSERT(shape.size() <= tensor_rank, "Unsupported shape rank of repacking data");
    init_parallel_domain(shape, tensor_rank, 2LU, dom);

    const auto& in_strides = input_repacker.in_offsets();
    const auto& out_strides = input_repacker.out_offsets();
    OPENVINO_ASSERT(everyone_is(tensor_rank, in_strides.size(), out_strides.size(), dom.size()),
                    "Unsupported shape rank of repacking data");

    const auto& kernel = input_repacker.kernel<BrgemmCopyBKernel>();
    if (tensor_rank == rank6D) {
        parallel4d_repacking(kernel.get(), dom, in_strides, out_strides, src_ptr, dst_ptr);
    } else {
        parallelNd_repacking(kernel.get(), dom, in_strides, out_strides, src_ptr, dst_ptr);
    }
}

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
// NOLINTBEGIN(misc-include-cleaner) bug in clang-tidy
void SubgraphExecutor::segfault_detector() const {
    static std::mutex err_print_lock;
    if (enabled_segfault_detector) {
        __sighandler_t signal_handler = []([[maybe_unused]] int signal) {
            std::lock_guard<std::mutex> guard(err_print_lock);
            if (auto* segfault_detector_emitter = ov::intel_cpu::g_custom_segfault_handler->local()) {
                std::cout << segfault_detector_emitter->info() << '\n';
            }
            auto tid = parallel_get_thread_num();
            OPENVINO_THROW("Segfault was caught by the signal handler in subgraph node execution on thread " +
                           std::to_string(tid));
        };
        struct sigaction new_handler {};
        new_handler.sa_handler = signal_handler;
        sigaction(SIGSEGV, &new_handler, nullptr);
    }
}
// NOLINTEND(misc-include-cleaner) bug in clang-tidy
#endif

std::vector<MemoryPtr> SubgraphExecutor::separately_repack_inputs(const dnnl::stream& strm,
                                                                  const std::vector<MemoryPtr>& src_mem_ptrs) {
    auto reordered_in_ptrs = src_mem_ptrs;
    size_t offset = m_internal_buffer_size;
    for (const auto& [in_idx, input_repacker] : m_input_repackers) {
        const auto& desc = input_repacker.desc();
        const void* data_ptr = m_buffer_scratchpad->getDataAs<uint8_t>() + offset;

        OPENVINO_ASSERT(in_idx < src_mem_ptrs.size(), "Incorrect index of input repacked mem ptr");
        const auto& src_mem = src_mem_ptrs[in_idx];
        const auto& dst_mem = std::make_shared<Memory>(strm.get_engine(), desc, data_ptr, false);
        separately_repack_input(src_mem, dst_mem, input_repacker, m_tensor_rank);

        reordered_in_ptrs[in_idx] = dst_mem;
        offset += desc->getCurrentMemSize();
    }
    return reordered_in_ptrs;
}

void SubgraphExecutor::in_parallel_repack_inputs(const std::vector<MemoryPtr>& in_mem_ptrs,
                                                 const std::vector<size_t>& indexes,
                                                 int ithr,
                                                 jit_snippets_call_args& call_args) {
    size_t repacked_offset_idx = 0;
    for (const auto& [in_idx, input_repacker] : m_input_repackers) {
        size_t src_offset = m_start_offset_in[in_idx];
        init_offset(input_repacker.in_offsets(), indexes, src_offset);

        auto* repacked_ptr = get_external_scratchpad_ptr(ithr, in_idx);

        auto& last_processed_src_offset = m_repacked_offsets_by_threads[ithr][repacked_offset_idx];
        if (src_offset != last_processed_src_offset) {
            BrgemmCopyBKernel::call_args args;
            args.src = in_mem_ptrs[in_idx]->getDataAs<const uint8_t>() + src_offset;
            args.tr_src = repacked_ptr;
            (*input_repacker.kernel<BrgemmCopyBKernel>())(&args);

            last_processed_src_offset = src_offset;
        }

        call_args.src_ptrs[in_idx] = repacked_ptr;
        ++repacked_offset_idx;
    }
}

void SubgraphExecutor::execute(const dnnl::stream& strm,
                               const std::vector<MemoryPtr>& in_mem_ptrs,
                               const std::vector<MemoryPtr>& out_mem_ptrs) {
    switch (get_repacking_impl_type()) {
    case RepackingImplType::SEPARATE:
        exec_impl(separately_repack_inputs(strm, in_mem_ptrs), out_mem_ptrs);
        return;
    case RepackingImplType::IN_PARALLEL:
    case RepackingImplType::NONE:
        exec_impl(in_mem_ptrs, out_mem_ptrs);
        return;
    default:
        OPENVINO_THROW("Uknown RepackingImplType");
    }
}

void SubgraphStaticExecutor::exec_impl(const std::vector<MemoryPtr>& in_mem_ptrs,
                                       const std::vector<MemoryPtr>& out_mem_ptrs) {
    const auto& callable = m_schedule->get_callable<kernel>();

    initializer_functor initializer;
    call_functor caller;

    switch (get_repacking_impl_type()) {
    case RepackingImplType::IN_PARALLEL:
        initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
            init_call_args(call_args, in_mem_ptrs, out_mem_ptrs, m_start_offset_in, m_start_offset_out);
            update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
            clean_repacked_offsets(ithr);
        };
        caller = [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, size_t ithr) {
            in_parallel_repack_inputs(in_mem_ptrs, indexes, ithr, call_args);
            callable(&call_args, indexes.data());
        };
        break;
    case RepackingImplType::SEPARATE:
    case RepackingImplType::NONE:
        initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
            init_call_args(call_args, in_mem_ptrs, out_mem_ptrs, m_start_offset_in, m_start_offset_out);
            update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
        };
        caller =
            [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, [[maybe_unused]] size_t ithr) {
                callable(&call_args, indexes.data());
            };
        break;
    default:
        OPENVINO_THROW("Uknown RepackingImplType");
    }

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    if (m_parallel_exec_domain.size() == rank6D) {
        parallel_for6d(initializer, caller);
    } else {
        parallel_forNd(initializer, caller);
    }
}

void SubgraphDynamicSpecializedExecutor::exec_impl(const std::vector<MemoryPtr>& in_mem_ptrs,
                                                   const std::vector<MemoryPtr>& out_mem_ptrs) {
    const auto& callable = m_schedule->get_callable<dynamic_kernel>();

    OPENVINO_ASSERT(m_data_offsets.size() == in_mem_ptrs.size() + out_mem_ptrs.size(), "Incorrect data offset count!");
    OPENVINO_ASSERT(m_data_offsets.front().size() == m_parallel_exec_domain.size(),
                    "Data offsets with invalid ranks detected");

    // Note: we need to reset KernelExecutorTable to the state that was recorded in the
    // SubgraphDynamicSpecializedExecutor constructor because the table might've been used for other shapes
    m_reset_exec_table_state();

    std::vector<const uint8_t*> src_ptrs;
    std::vector<uint8_t*> dst_ptrs;
    init_original_ptrs(in_mem_ptrs, out_mem_ptrs, src_ptrs, dst_ptrs, m_start_offset_in, m_start_offset_out);

    initializer_functor initializer;
    call_functor caller;

    switch (get_repacking_impl_type()) {
    case RepackingImplType::IN_PARALLEL:
        initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
            init_call_args(call_args);
            update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
            clean_repacked_offsets(ithr);
        };
        caller = [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, size_t ithr) {
            update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
            in_parallel_repack_inputs(in_mem_ptrs, indexes, ithr, call_args);
            callable(&call_args);
        };
        break;
    case RepackingImplType::SEPARATE:
    case RepackingImplType::NONE:
        initializer = [&](jit_snippets_call_args& call_args, size_t ithr) {
            init_call_args(call_args);
            update_scratchpad_ptr(call_args.buffer_scratchpad_ptr, ithr);
        };
        caller =
            [&](jit_snippets_call_args& call_args, const std::vector<size_t>& indexes, [[maybe_unused]] size_t ithr) {
                update_ptrs(call_args, src_ptrs, dst_ptrs, indexes);
                callable(&call_args);
            };
        break;
    default:
        OPENVINO_THROW("Uknown RepackingImplType");
    }

#if defined(__linux__) && defined(SNIPPETS_DEBUG_CAPS)
    segfault_detector();
#endif

    if (m_parallel_exec_domain.size() == rank6D) {
        parallel_for6d(initializer, caller);
    } else {
        parallel_forNd(initializer, caller);
    }
}

}  // namespace ov::intel_cpu
