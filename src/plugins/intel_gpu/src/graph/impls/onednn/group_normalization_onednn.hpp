// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization_inst.h"
#include "impls/onednn/utils.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"
#include "intel_gpu/runtime/utils.hpp"
#include "impls/registry/implementation_manager.hpp"

#include <memory>
#include <cmath>

namespace cldnn {
namespace onednn {

struct GroupNormalizationImplementationManager : public ImplementationManager {
    OV_GPU_PRIMITIVE_IMPL("onednn::gn")
//    GroupNormalizationImplementationManager(shape_types shape_type, ValidateFunc vf = nullptr) : ImplementationManager(impl_types::onednn, shape_type, vf) {}
    GroupNormalizationImplementationManager(shape_types shape_type) : ImplementationManager(impl_types::onednn, shape_type) {}
    std::unique_ptr<primitive_impl> create_impl(const program_node& node, const kernel_impl_params& params) const override;

    bool validate_impl(const program_node& node) const override {
        assert(node.is_type<group_normalization>());
        const auto& info = node.get_program().get_engine().get_device_info();
        if (!info.supports_immad || info.arch == gpu_arch::unknown)
            return false;

        const auto& gn_node = node.as<group_normalization>();
        const auto& in_layout = gn_node.get_input_layout(0);
        const auto& out_layout = gn_node.get_output_layout(0);
        auto in0_dt = in_layout.data_type;
        auto gn_prim = gn_node.get_primitive();

        /*
        auto types = {data_types::f16, data_types::f32};
        static const std::vector<format::type> supported_formats = {
          format::any
          format::bfyx,
          format::b_fs_yx_fsv16,
        };
        */

        if (one_of(data_types::i64, {in0_dt}))
            return false;

        if (!everyone_is(format::bfyx, in_layout.format, out_layout.format) && !everyone_is(format::any, in_layout.format, out_layout.format))
            return false;

        if (!is_supported_pad(in_layout) || !is_supported_pad(out_layout))
            return false;

//        const auto& output_layout = gn_node.get_output_layout();
//        const auto& ps = output_layout.get_partial_shape();

        return true;
    }

    in_out_fmts_t query_formats(const program_node& node) const override {
        assert(node.is_type<group_normalization>());
        printf("query format come into here\n");
        std::vector<format::type> in_fmts(node.get_dependencies().size(), format::any);
        std::vector<format::type> out_fmts(node.get_outputs_count(), format::any);


        size_t out_rank = node.get_output_layout().get_rank();
        for (size_t idx = 0 ; idx < node.get_dependencies().size() ; idx++) {
            if (node.get_dependency(idx).is_constant())
                continue;

            auto target_format = format::get_default_format(out_rank);

            in_fmts[idx] = target_format;
        }
        out_fmts[0] = format::get_default_format(out_rank);

        return {in_fmts, out_fmts};
    }
};

}  // namespace onednn
}  // namespace cldnn
