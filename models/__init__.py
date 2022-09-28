#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.abs

from models.base_ssl3d_model import BaseSSLMultiInputOutputModel

def build_model(model_config, cluster, logger, linear_probe = False):
    return BaseSSLMultiInputOutputModel(model_config, cluster, logger, linear_probe)


__all__ = ["BaseSSLMultiInputOutputModel", "build_model"]
