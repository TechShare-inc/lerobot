#!/usr/bin/env python

from contextlib import nullcontext

import pytest
import torch

pytest.importorskip("datasets", reason="datasets is required (install lerobot[dataset])")

from lerobot.scripts.lerobot_train import update_policy  # noqa: E402
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker  # noqa: E402


class DummyAccelerator:
    num_processes = 1

    def __init__(self):
        self.no_sync_calls = 0

    def autocast(self):
        return nullcontext()

    def backward(self, loss):
        loss.backward()

    def clip_grad_norm_(self, parameters, grad_clip_norm):
        return torch.nn.utils.clip_grad_norm_(parameters, grad_clip_norm)

    def no_sync(self, model):
        self.no_sync_calls += 1
        return nullcontext()

    def unwrap_model(self, model, keep_fp32_wrapper=True):
        return model


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(self.linear.weight)

    def forward(self, batch):
        pred = self.linear(batch["x"])
        loss = torch.nn.functional.mse_loss(pred, batch["y"])
        return loss, {"micro_loss": loss.item()}


class StepCountingScheduler:
    def __init__(self):
        self.step_calls = 0

    def step(self):
        self.step_calls += 1


def test_update_policy_accumulates_micro_batches_before_optimizer_step():
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    scheduler = StepCountingScheduler()
    accelerator = DummyAccelerator()
    metric_meters = {
        "loss": AverageMeter("loss"),
        "grad_norm": AverageMeter("grdn"),
        "lr": AverageMeter("lr"),
        "update_s": AverageMeter("updt_s"),
    }
    if torch.cuda.is_available():
        metric_meters["gpu_mem_gb"] = AverageMeter("mem_gb")

    metrics = MetricsTracker(
        batch_size=2,
        num_frames=10,
        num_episodes=1,
        metrics=metric_meters,
        accelerator=accelerator,
    )

    step_calls = 0
    original_step = optimizer.step

    def step_once(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    optimizer.step = step_once

    update_policy(
        metrics,
        policy,
        [
            {"x": torch.tensor([[1.0]]), "y": torch.tensor([[1.0]])},
            {"x": torch.tensor([[1.0]]), "y": torch.tensor([[3.0]])},
        ],
        optimizer,
        grad_clip_norm=0.0,
        accelerator=accelerator,
        lr_scheduler=scheduler,
        gradient_accumulation_steps=2,
    )

    assert step_calls == 1
    assert scheduler.step_calls == 1
    assert accelerator.no_sync_calls == 1
    assert metrics.loss.val == pytest.approx(5.0)
    assert policy.linear.weight.item() == pytest.approx(0.4)
