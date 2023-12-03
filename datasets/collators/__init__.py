#!/usr/bin/env python3
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from datasets.collators.moco_collator import moco_collator
# from datasets.collators.sparse_collator import sparse_moco_collator
from datasets.collators.downstream_collator import downstream_collator

COLLATORS_MAP = {
    "moco_collator": moco_collator,
    "downstream_collator": downstream_collator,
    # "sparse_moco_collator": sparse_moco_collator
}


def get_collator(name):
    assert name in COLLATORS_MAP, "Unknown collator"
    return COLLATORS_MAP[name]


__all__ = ["get_collator"]
