// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/reference/detection_output.hpp"

#include "evaluate_node.hpp"
#include "openvino/core/type/element_type_traits.hpp"

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v0::DetectionOutput>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                         op->get_input_shape(0),
                                                         op->get_input_shape(1),
                                                         op->get_input_shape(2),
                                                         op->get_output_shape(0));
    if (op->get_input_size() == 3) {
        refDetOut.run(inputs[0].data<const T>(),
                      inputs[1].data<const T>(),
                      inputs[2].data<const T>(),
                      nullptr,
                      nullptr,
                      outputs[0].data<T>());
    } else if (op->get_input_size() == 5) {
        refDetOut.run(inputs[0].data<const T>(),
                      inputs[1].data<const T>(),
                      inputs[2].data<const T>(),
                      inputs[3].data<const T>(),
                      inputs[4].data<const T>(),
                      outputs[0].data<T>());
    } else {
        OPENVINO_THROW("DetectionOutput layer supports only 3 or 5 inputs");
    }
    return true;
}

template <ov::element::Type_t ET>
bool evaluate(const std::shared_ptr<ov::op::v8::DetectionOutput>& op,
              ov::TensorVector& outputs,
              const ov::TensorVector& inputs) {
    using T = typename ov::element_type_traits<ET>::value_type;
    ov::reference::referenceDetectionOutput<T> refDetOut(op->get_attrs(),
                                                         op->get_input_shape(0),
                                                         op->get_input_shape(1),
                                                         op->get_input_shape(2),
                                                         op->get_output_shape(0));
    if (op->get_input_size() == 3) {
        refDetOut.run(inputs[0].data<const T>(),
                      inputs[1].data<const T>(),
                      inputs[2].data<const T>(),
                      nullptr,
                      nullptr,
                      outputs[0].data<T>());
    } else if (op->get_input_size() == 5) {
        refDetOut.run(inputs[0].data<const T>(),
                      inputs[1].data<const T>(),
                      inputs[2].data<const T>(),
                      inputs[3].data<const T>(),
                      inputs[4].data<const T>(),
                      outputs[0].data<T>());
    } else {
        OPENVINO_THROW("DetectionOutput layer supports only 3 or 5 inputs");
    }
    return true;
}

template <>
bool evaluate_node<ov::op::v0::DetectionOutput>(std::shared_ptr<ov::Node> node,
                                                ov::TensorVector& outputs,
                                                const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v0::DetectionOutput>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v0::DetectionOutput>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v0::DetectionOutput>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v0::DetectionOutput>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}

template <>
bool evaluate_node<ov::op::v8::DetectionOutput>(std::shared_ptr<ov::Node> node,
                                                ov::TensorVector& outputs,
                                                const ov::TensorVector& inputs) {
    const auto& element_type = node->get_output_element_type(0);

    switch (element_type) {
    case ov::element::bf16:
        return evaluate<ov::element::bf16>(ov::as_type_ptr<ov::op::v8::DetectionOutput>(node), outputs, inputs);
    case ov::element::f16:
        return evaluate<ov::element::f16>(ov::as_type_ptr<ov::op::v8::DetectionOutput>(node), outputs, inputs);
    case ov::element::f64:
        return evaluate<ov::element::f64>(ov::as_type_ptr<ov::op::v8::DetectionOutput>(node), outputs, inputs);
    case ov::element::f32:
        return evaluate<ov::element::f32>(ov::as_type_ptr<ov::op::v8::DetectionOutput>(node), outputs, inputs);
    default:
        OPENVINO_THROW("Unhandled data type ", node->get_element_type().get_type_name(), " in evaluate_node()");
    }
}
