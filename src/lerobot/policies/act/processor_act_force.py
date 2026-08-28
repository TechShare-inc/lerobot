#!/usr/bin/env python

from typing import Any

import torch

from lerobot.processor import PolicyAction, PolicyProcessorPipeline

from .configuration_act_force import ACTForceConfig
from .processor_act import make_act_pre_post_processors


def make_act_force_pre_post_processors(
    config: ACTForceConfig,
    dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
) -> tuple[
    PolicyProcessorPipeline[dict[str, Any], dict[str, Any]],
    PolicyProcessorPipeline[PolicyAction, PolicyAction],
]:
    """Creates ACTForce processors.

    Force features use the same normalization path as other ACT numeric observations.
    """

    return make_act_pre_post_processors(config, dataset_stats=dataset_stats)
