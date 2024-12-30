// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "group_normalization_onednn.hpp"
#include "group_normalization_inst.h"
//#include "intel_gpu/primitives/group_normalization.hpp"
//#include "intel_gpu/runtime/utils.hpp"
#include "primitive_onednn_base.h"
#include "impls/registry/implementation_manager.hpp"

#include <oneapi/dnnl/dnnl.hpp>

//#include <algorithm>
#include <memory>
//#include <cmath>
namespace cldnn {
namespace onednn {

struct group_normalization_onednn : typed_primitive_onednn_impl<group_normalization> {
    using parent = typed_primitive_onednn_impl<group_normalization>;
    using parent::parent;

    DECLARE_OBJECT_TYPE_SERIALIZATION(cldnn::onednn::group_normalization_onednn)

private:
    int _ds_group_size;
    dnnl::memory::data_type _ds_data_type;

    static std::vector<int64_t> reshape_to_2d(const ov::PartialShape& shape, int64_t feature) {
        auto staticShape = shape.to_shape();
        size_t total =
            std::accumulate(staticShape.begin(), staticShape.end(), static_cast<size_t>(1), std::multiplies<size_t>());
        std::vector<int64_t> reshapeSize = { static_cast<int64_t>(total) / feature, feature };
        return reshapeSize;
    }

protected:
    std::unique_ptr<primitive_impl> clone() const override {
        return make_unique<group_normalization_onednn>(*this);
    }

    std::unordered_map<int, dnnl::memory> get_arguments(group_normalization_inst& instance) const override {
        std::unordered_map<int, dnnl::memory> args = parent::get_arguments(instance);

        {
            auto& input = instance.input_memory(0);
            auto offset = onednn::get_offset(instance.get_input_layout(0), _pd.dnnl::primitive_desc_base::src_desc(0));
            args.insert({DNNL_ARG_SRC, input.get_onednn_memory(_pd.dnnl::primitive_desc_base::src_desc(0), offset)});
        }

        {
            auto& scale = instance.input_memory(1);
            auto offset = onednn::get_offset(instance.get_input_layout(1), _pd.dnnl::primitive_desc_base::weights_desc(0));
            args.insert({DNNL_ARG_SCALE, scale.get_onednn_memory(_pd.weights_desc(0), offset)});
        }

        {
            auto& bias = instance.input_memory(2);
            auto offset = onednn::get_offset(instance.get_input_layout(2), _pd.dnnl::primitive_desc_base::weights_desc(0));
            args.insert({DNNL_ARG_SHIFT, bias.get_onednn_memory(_pd.weights_desc(0), offset)});
        }

        {
            auto& output = instance.output_memory();
            auto offset = onednn::get_offset(instance.get_output_layout(), _pd.dnnl::primitive_desc_base::dst_desc(0));
            args.insert({DNNL_ARG_DST, output.get_onednn_memory(_pd.dnnl::primitive_desc_base::dst_desc(0), offset)});
        }

        return args;
    }

    static std::shared_ptr<dnnl::group_normalization_forward::primitive_desc>
        get_group_normalization_primitive_descriptor(const kernel_impl_params& impl_params,
                                        cldnn::engine& engine,
                                        const dnnl::primitive_attr& attr = dnnl::primitive_attr()) {
        auto input_layout = impl_params.get_input_layout(0);
        auto output_layout = impl_params.get_output_layout();
        auto prim = impl_params.typed_desc<group_normalization>();

        auto input_md = onednn::layout_to_memory_desc(input_layout);
        auto output_md = onednn::layout_to_memory_desc(output_layout);

        return std::make_shared<dnnl::group_normalization_forward::primitive_desc>(
                engine.get_onednn_engine(),
                dnnl::prop_kind::forward_inference,
                input_md,
                output_md,
                prim->num_groups,
                prim->epsilon,
                dnnl::normalization_flags::use_scale | dnnl::normalization_flags::use_shift);
    }

public:
    void save(BinaryOutputBuffer& ob) const override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::save(ob);


        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ob.getKernelImplParams());
        auto prim = impl_params->typed_desc<group_normalization>();
        //size_t input_size = prim->input_size;
        //bool has_bias = !prim->bias.empty();
        //ob << input_size;
        //ob << has_bias;

        std::vector<uint8_t> prim_cache;
        prim_cache = _prim.get_cache_blob();
        ob << prim_cache;
#endif
    }

    void load(BinaryInputBuffer& ib) override {
#ifdef ONEDNN_PRIMITIVE_SERIALIZATION
        parent::load(ib);

        //size_t input_size = 2;
        //bool has_bias = false;
        //ib >> input_size;
        //ib >> has_bias;

        const kernel_impl_params* impl_params = reinterpret_cast<kernel_impl_params*>(ib.getKernelImplParams());
        auto prim = impl_params->typed_desc<group_normalization>();

        auto prim_desc = get_group_normalization_primitive_descriptor(*impl_params, ib.get_engine(), *_attrs);
        _pd = *prim_desc;

        std::vector<uint8_t> prim_cache;
        ib >> prim_cache;

        _scratchpad_md = _pd.scratchpad_desc();

        _prim = dnnl::primitive(_pd, prim_cache);
#endif
    }

    static std::unique_ptr<primitive_impl> create(const group_normalization_node& arg, const kernel_impl_params& impl_params) {
        auto& engine = impl_params.prog->get_engine();
        auto& config = impl_params.prog->get_config();
        auto attr = impl_params.attrs_onednn;
        auto prim = impl_params.typed_desc<group_normalization>();

        auto prim_desc = get_group_normalization_primitive_descriptor(impl_params, impl_params.prog->get_engine(), *attr);
        return cldnn::make_unique<group_normalization_onednn>(engine, config, attr, *prim_desc);
    }
};

std::unique_ptr<primitive_impl> GroupNormalizationImplementationManager::create_impl(const program_node& node, const kernel_impl_params& params) const {
    assert(node.is_type<group_normalization>());
    return onednn::group_normalization_onednn::create(static_cast<const group_normalization_node&>(node), params);
}

}  // namespace onednn
}  // namespace cldnn

BIND_BINARY_BUFFER_WITH_TYPE(cldnn::onednn::group_normalization_onednn)
