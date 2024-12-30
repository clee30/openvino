// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "impls/registry/predicates.hpp"
#include "primitive_inst.h"
#include "registry.hpp"
#include "intel_gpu/primitives/group_normalization.hpp"


#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/group_normalization_onednn.hpp"
#endif

#if OV_GPU_WITH_CM
    #include "impls/cm/impl_example.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<group_normalization>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
//        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::GroupNormalizationImplementationManager, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::GroupNormalizationImplementationManager, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(group_normalization, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(group_normalization, shape_types::dynamic_shape,
            [](const program_node& node) {
                if (node.can_use(impl_types::onednn))
                    return false;
                return node.get_output_pshape().size() <= 3;
        })
        OV_GPU_CREATE_INSTANCE_CM(cm::ExampleImplementationManager, shape_types::static_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
