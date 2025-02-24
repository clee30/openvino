# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import re
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from logging import getLogger

import numpy as np
import pytest

from e2e_tests.test_utils.path_utils import resolve_file_path
# import local modules:
from e2e_tests.test_utils.test_utils import align_output_name
from e2e_tests.common.parsers import mapping_parser as mapping
from e2e_tests.common.common.e2e_utils import get_tensor_names_dict
from e2e_tests.test_utils.env_tools import Environment

log = getLogger(__name__)


def parse_mo_mapping(mo_out, model_name):
    """
    Parse model optimizer mapping file given output dir and model name.

    This is the basic function that provides mapping attribute for
    CommonConfig class.

    :param mo_out:  model optimizer output directory
    :param model_name:  model name (i.e. alexnet.pb for TF, alexnet.caffemodel
                        for Caffe)
    :return:    model optimizer mapping dictionary with fw layer names as keys
                and ir layer names as values
    """
    model_base_name = os.path.splitext(model_name)[0]
    mapping_file = os.path.join(mo_out, model_base_name + ".mapping")
    return mapping(resolve_file_path(mapping_file, as_str=True))


class CommonConfig:
    """
    Base class for E2E test classes. Provides class-level method to align
    reference and IE results.

    :attr mapping:  dict-like entity that maps framework (e.g. TensorFlow)
                        model layers to optimized model (processed by model
                        optimizer) layers
    :attr model:    model name used to detect mapping file if not specified
                        with mapping argument
    :attr use_mo_mapping:   specifies if should use MO mapping file, one can
                                override the value in test subclass to control
                                the behavior
    """
    mapping = None
    use_mo_mapping = True
    convert_pytorch_to_onnx = None
    __pytest_marks__ = tuple([
        pytest.mark.api_enabling,
        pytest.mark.components("openvino.test:e2e_tests"),
    ])

    def __new__(cls, test_id, *args, **kwargs):
        """Specifies all required fields for a test instance"""
        instance = super().__new__(cls)
        instance.test_id = test_id
        instance.required_params = {}
        for param_name, param_val in kwargs.items():
            if not hasattr(instance, param_name):
                setattr(instance, param_name, param_val)
        # Every test instance manages it's own environment. To make tests process-safe, output directories
        # are redirected to a subdirectory unique for each test.
        instance.environment = Environment.env.copy()
        subpath = re.sub(r'[^\w\-_\. ]', "_", test_id)   # filter all symbols not supported in a file systems
        tmpdir_subpath = Path(TemporaryDirectory(prefix=subpath).name).name
        for env_key in ["mo_out", "pytorch_to_onnx_dump_path", "pregen_irs_path"]:
            instance.environment[env_key] = str(Path(instance.environment[env_key]) / tmpdir_subpath)
        return instance

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls, self.test_id)
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, deepcopy(key, memo), deepcopy(value, memo))
        return result

    def prepare_prerequisites(self, *args, **kwargs):
        """
        Prepares prerequisites required for tests: download models, references etc.
        Function also may fill instance's fields.
        """
        pass

    def align_results(self, ref_res, optim_model_res, xml=None):
        """
        Aligns optimized model results with reference model results.

        This is achieved by changing optimized model result keys (corresponding
        to output layers) to framework model results names according to
        mapping attribute.

        When use_mo_mapping is False, no alignment is performed.

        If mapping is not provided, it is deduced from model attribute.

        If mapping and model both not set, no alignment is performed.

        :param ref_res: reference model results
        :param optim_model_res:
        :param xml: XML file generated by MO
        :return: aligned results (ref_res, optim_model_res) with same keys
        """

        log.debug(f"Aligning results")
        log.debug(f"ref_res.keys() {ref_res.keys()}")
        log.debug(f"optim_model_res.keys() {optim_model_res.keys()}")
        if len(ref_res) == 1 and len(optim_model_res) == 1:
            ref_res_vals = list(ref_res.values())[0]
            ie_res_vals = list(optim_model_res.values())[0]
            if (isinstance(ref_res_vals, np.ndarray) and isinstance(
                    ie_res_vals, np.ndarray)) and ref_res_vals.shape == ie_res_vals.shape:
                ref_layer_name = next(iter(ref_res.keys()))
                optim_model_res = {ref_layer_name: ie_res_vals}
                ref_res = {ref_layer_name: ref_res_vals}
                return ref_res, optim_model_res

        if not self.use_mo_mapping:
            return ref_res, optim_model_res

        if ref_res.keys() == optim_model_res.keys():
            return ref_res, optim_model_res

        if not self.mapping:
            log.debug(f"Aligning results using mapping")
            pre_generated_irs = self.ie_pipeline.get('get_ir').get('pregenerated')
            if pre_generated_irs:
                log.info("Construct mapping attribute from pre-generated IRs")
                xml_file = pre_generated_irs.get('xml')
                resolved_path = resolve_file_path(xml_file, as_str=True)
                self.mapping = get_tensor_names_dict(xml_ir=resolved_path)
            elif not pre_generated_irs:
                resolved_path = resolve_file_path(xml, as_str=True)
                self.mapping = get_tensor_names_dict(xml_ir=resolved_path)
            else:
                error = f"{self.__class__.__name__} should use 'model' or 'model_path' attribute to define model"
                raise Exception(error)

        missed_ir_layer_names = []
        missed_fw_layer_names = []
        not_contain_layers_in_mapping_err_msg = ''
        not_found_layers_in_inference_err_msg = ''
        for fw_layer_name in ref_res.keys():
            if fw_layer_name not in optim_model_res.keys():
                aligned_name = align_output_name(fw_layer_name, optim_model_res.keys())
                ir_layer_name = self.mapping.get(fw_layer_name, None)

                # WA for CVS-94674
                if isinstance(ir_layer_name, list):
                    for name in ir_layer_name:
                        for ov_name in optim_model_res.keys():
                            if name == ov_name:
                                ir_layer_name = ov_name
                                break
                    if isinstance(ir_layer_name, list):
                        raise Exception(f"Output tensor names in references and in ov model are different\nRef names: "
                                        f"{ref_res.keys()}\nOV names: {optim_model_res.keys()}")

                if not ir_layer_name and not aligned_name:
                    missed_fw_layer_names.append(fw_layer_name)
                    continue
                if aligned_name:
                    optim_model_res[fw_layer_name] = optim_model_res.pop(aligned_name)
                    continue
                if ir_layer_name not in optim_model_res:
                    missed_ir_layer_names.append(ir_layer_name)
                    continue
                optim_model_res[fw_layer_name] = optim_model_res.pop(ir_layer_name)

        if missed_fw_layer_names:
            not_contain_layers_in_mapping_err_msg = 'mapping file does not contain {fw_layer}. Mapping: {mapping}'\
                .format(fw_layer=missed_fw_layer_names, mapping=self.mapping)
        if missed_ir_layer_names:
            not_found_layers_in_inference_err_msg = 'found IR layer {ir_layer} is not found in inference result. '\
                                                    'available layers: {avail_layers}. Mapping: {mapping}' \
                                                    .format(ir_layer=missed_ir_layer_names,
                                                            avail_layers=optim_model_res.keys(),
                                                            mapping=self.mapping)
        if not_contain_layers_in_mapping_err_msg or not_found_layers_in_inference_err_msg:
            raise ValueError('{}\n{}'.format(not_contain_layers_in_mapping_err_msg,
                                             not_found_layers_in_inference_err_msg))
        return ref_res, optim_model_res

    def _add_defect(self, name, condition, params, test_name=None):
        self.__pytest_marks__ += tuple([
            pytest.mark.bugs(
                name,
                condition,
                params,
                test_name
            )]
        )

    def _set_test_group(self, name, condition=True, params=None, test_name=None):
        mark = pytest.mark.test_group(name, condition, params, test_name)

        # Note: it is possible that other test groups are already in __pytest_marks__,
        # so we wish to resolve inserted mark prior any existing test_group marks.
        self.__pytest_marks__ = (mark, ) + self.__pytest_marks__    # add mark as first element in tuple.
