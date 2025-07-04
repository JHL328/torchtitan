# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


# Import the built-in models here so that the corresponding register_model_spec()
# will be called.
import torchtitan.models.llama  # noqa: F401


model_name_to_tokenizer = {
    "llama3": "tiktoken",
    "llama3_nsa": "tiktoken"
}
