# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tarfile
from contextlib import nullcontext
from typing import Callable, Optional

import torch
import torch.distributed as dist
from megatron.core import mpu, parallel_state
from megatron.core.transformer.module import Float16Module
from omegaconf import OmegaConf
from omegaconf.omegaconf import DictConfig, open_dict

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.utils_funcs import torch_dtype_from_precision
from nemo.utils import logging
from nemo.utils.distributed import temporary_directory
from nemo.utils.model_utils import save_artifacts, unwrap_model

try:
    import modelopt.torch.quantization as mtq
    from modelopt.torch.export import export_tensorrt_llm_checkpoint
    from modelopt.torch.utils.distributed import set_data_parallel_group, set_tensor_parallel_group

    HAVE_MODELOPT = True

except (ImportError, ModuleNotFoundError) as e:
    HAVE_MODELOPT = False
    HAVE_MODELOPT_ERROR = e


SUPPORTED_DTYPE = [16, "16", "bf16"]  # Default precision for non-quantized layers
QUANT_CFG_CHOICES = {
    "int8": mtq.INT8_DEFAULT_CFG,
    "int8_sq": mtq.INT8_SMOOTHQUANT_CFG,
    "fp8": mtq.FP8_DEFAULT_CFG,
    "int4_awq": mtq.INT4_AWQ_CFG,
    "w4a8_awq": mtq.W4A8_AWQ_BETA_CFG,
    "int4": mtq.INT4_BLOCKWISE_WEIGHT_ONLY_CFG,
}


class Quantizer:
    """
    Post-training quantization (PTQ) and TRT-LLM export of Nemo checkpoints.

    PTQ converts selected model layers to low-precision format (e.g., INT4, FP8) for efficient serving.
    The process consist of several steps:

        1. Loading a Nemo model from disk using appropriate parallelism strategy
        2. Calibrating the model to obtain appropriate algorithm-specific scaling factors
        3. Producing output directory or .qnemo tarball with model config (json),
           quantized weights (safetensors) and tokenizer config (yaml).

    The output directory (or .qnemo file) produced is intended to be consumed by TensorRT-LLM toolbox
    for efficient inference. This can be achieved using Nemo inference containers.

    Currently supported and tested model family is Llama2. Model type needs to be specified in
    the quantization command with decoder_type parameter on exporting (see below). Quantizing other
    model families is experimental and might not be fully supported.

    Available quantization methods are listed in `QUANT_CFG_CHOICES` dictionary above.
    Please consult Model Optimizer documentation https://nvidia.github.io/TensorRT-Model-Optimizer/ for details.
    You can also inspect different choices in examples/nlp/language_modeling/conf/megatron_quantization.yaml
    for quantization algorithms and calibration data as well as recommended settings.

    Quantization algorithm can also be conveniently set to 'null' to perform only weights export step
    for TensorRT-LLM deployment. This is useful to getting baseline results for a full-precision model.
    """

    @staticmethod
    def _setup(model: MegatronGPTModel, inference_config: Optional[DictConfig] = None):
        """Setup model for quantization."""
        try:
            model.model.module.language_model.encoder.activations_checkpoint_method = None
        except AttributeError:
            pass

        if not parallel_state.is_initialized():

            def dummy():
                return

            if model.trainer.strategy.launcher is not None:
                model.trainer.strategy.launcher.launch(dummy, trainer=model.trainer)
            model.trainer.strategy.setup_environment()

        set_data_parallel_group(mpu.get_data_parallel_group())
        set_tensor_parallel_group(mpu.get_tensor_model_parallel_group())

        if inference_config is not None:
            model.set_inference_config(OmegaConf.to_container(inference_config))

    @staticmethod
    def modify_model_config(model_cfg: DictConfig) -> DictConfig:
        """Modify model config for quantization."""
        with open_dict(model_cfg):
            model_cfg.activations_checkpoint_method = None
            model_cfg.activations_checkpoint_granularity = None
            model_cfg.sequence_parallel = False
            # Only custom ModelOpt spec is supported for Quantization: this custom spec is largely based on local Megatron-LM
            # layer definitions to avoid Transformer Engine implementations that are currently not supported.
            # This layer spec also requires RoPE fusion to be disabled for tensor view operations in attention
            # layer implementation from megatron/core/transformer/dot_product_attention.py to be functional.
            model_cfg.name = "modelopt"
            model_cfg.apply_rope_fusion = False

        return model_cfg

    @staticmethod
    def _sample_output(model: MegatronGPTModel):
        """Generate sample output for a model instance."""
        logging.info("Generating sample output for the model...")

        response = model.generate(
            inputs=[
                "Born in north-east France, Soyer trained as a",
                "Born in California, Soyer trained as a",
            ],
            length_params={
                "max_length": 100,
                "min_length": 100,
            },
        )

        logging.info(f'Example NeMo output before export: {response["sentences"]}"')

    @staticmethod
    def quantize(
        model: MegatronGPTModel,
        forward_loop: Callable[[MegatronGPTModel], None],
        quantization_config: DictConfig,
        inference_config: Optional[DictConfig] = None,
    ):
        """Quantize model checkpoint using given dataloader.

        Expected keys in `quantization_config`:
            - algorithm: str
            - decoder_type: str
            - awq_block_size: int (only for awq algorithms)
        """
        if not HAVE_MODELOPT:
            raise RuntimeError("nvidia-modelopt is needed to use Quantizer") from HAVE_MODELOPT_ERROR

        Quantizer._setup(model, inference_config)

        algorithm = quantization_config.algorithm
        assert algorithm in QUANT_CFG_CHOICES
        quant_cfg = QUANT_CFG_CHOICES[algorithm]
        decoder_type = quantization_config.decoder_type

        logging.info(f"Quantizing model to {algorithm}...")

        if "awq" in algorithm:
            weight_quantizer = quant_cfg["quant_cfg"]["*weight_quantizer"]
            if isinstance(weight_quantizer, list):
                weight_quantizer = weight_quantizer[0]
            weight_quantizer["block_sizes"][-1] = quantization_config.awq_block_size

        # Always turn on FP8 kv cache to save memory footprint.
        # For int8_sq, we use int8 kv cache.
        # TODO: Investigate why enabling FP8 kv cache will cause accuracy regressions for Nemotron.
        enable_quant_kv_cache = "int8" not in algorithm and decoder_type != "gptnext"
        logging.info(f'{"Enabled" if enable_quant_kv_cache else "Disabled"} KV cache quantization')
        quant_cfg["quant_cfg"]["*output_quantizer"] = {
            "num_bits": 8 if algorithm == "int8_sq" else (4, 3),
            "axis": None,
            "enable": enable_quant_kv_cache,
        }

        model = mtq.quantize(model, quant_cfg, forward_loop)

        if decoder_type == "gptnext":
            # We found squared_relu may have an under-calibration problem.
            # Clamp the scaling_factor with a min threshold to avoid under-calibration.
            maxbound = 0
            if algorithm == "fp8":
                maxbound = 448
            elif algorithm == "int8_sq":
                maxbound = 127
            model = mtq.postprocess_amax(
                model, "*input_quantizer", lambda amax: torch.clamp(amax, min=0.01 * maxbound)
            )

        if dist.get_rank() == 0:
            mtq.print_quant_summary(model)

        return model

    @staticmethod
    def export(model: MegatronGPTModel, export_config: DictConfig):
        """Export model to '.qnemo' format for TensorRT-LLM engine build.

        Expected keys in `export_config`:
            - dtype: str/int
            - decoder_type: str
            - inference_tensor_parallel: int
            - inference_pipeline_parallel: int
            - save_path: str
        """
        if not HAVE_MODELOPT:
            raise RuntimeError("nvidia-modelopt is needed to use Quantizer") from HAVE_MODELOPT_ERROR

        assert export_config.dtype in SUPPORTED_DTYPE

        torch_dtype = torch_dtype_from_precision(export_config.dtype)

        Quantizer._sample_output(model)

        if model.cfg.megatron_amp_O2:
            model.model = unwrap_model(model.model, Float16Module)

        # Setup model export handling: temporary directory for
        # '.qnemo' tarball or directly write to export_config.save_path
        save_qnemo = export_config.save_path.endswith(".qnemo")
        if save_qnemo:
            export_handler = temporary_directory()
        else:
            export_handler = nullcontext(enter_result=export_config.save_path)

        with export_handler as export_dir:
            export_tensorrt_llm_checkpoint(
                model=model,
                decoder_type=export_config.decoder_type,
                dtype=torch_dtype,
                export_dir=export_dir,
                inference_tensor_parallel=export_config.inference_tensor_parallel,
                inference_pipeline_parallel=export_config.inference_pipeline_parallel,
                use_nfs_workspace=export_config.inference_pipeline_parallel == 1
                and model.cfg.pipeline_model_parallel_size > 1,
            )
            dist.barrier()  # Wait until all ranks complete export_model_config step
            logging.info(
                f"Exporting quantized weights, model artifacts, and tokenizer config to {export_config.save_path}..."
            )
            if dist.get_rank() == 0:
                save_artifacts(model, export_dir)
                if save_qnemo:
                    with tarfile.open(export_config.save_path, "w:gz") as tar:
                        tar.add(export_dir, arcname="./")
