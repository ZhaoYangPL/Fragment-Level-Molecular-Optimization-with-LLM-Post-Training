from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from tinker import types


MODULE_DIR = Path(__file__).resolve().parents[1]
CORE_PATH = MODULE_DIR / "TURN2Bonus0.25ExplicitKL_delta_abs_rl_core.py"
sys.path.insert(0, str(MODULE_DIR))

SPEC = importlib.util.spec_from_file_location("delta_nte_explicit_kl_core", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load training core from {CORE_PATH}")
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def _tensor_data(values, dtype):
    return types.TensorData.from_numpy(np.asarray(values, dtype=dtype))


def _datum(*, target_tokens, old_logprobs, advantages):
    return types.Datum(
        model_input=types.ModelInput.from_ints(tokens=[10, 11]),
        loss_fn_inputs={
            "target_tokens": _tensor_data(target_tokens, np.int64),
            "logprobs": _tensor_data(old_logprobs, np.float32),
            "advantages": _tensor_data(advantages, np.float32),
        },
    )


class _ReferenceClient:
    def __init__(self, logprobs):
        self.logprobs = logprobs

    async def compute_logprobs_async(self, _sequence_input):
        return list(self.logprobs)


class ExplicitKlTests(unittest.TestCase):
    def test_forward_kl_zero_and_gradient_is_finite(self):
        new_logprobs = torch.tensor([-0.2, -0.4], requires_grad=True)
        loss, metrics = core.compute_ppo_forward_kl_loss(
            [new_logprobs],
            old_logprobs_list=[[-0.2, -0.4]],
            reference_logprobs_list=[[-0.2, -0.4]],
            advantages_list=[[1.0, -0.5]],
            action_masks=[[1.0, 1.0]],
            kl_penalty_coef=0.03,
        )

        self.assertAlmostEqual(metrics["explicit_forward_kl"], 0.0, places=7)
        self.assertTrue(math.isfinite(metrics["ppo_kl_total_loss"]))
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(new_logprobs.grad)))

    def test_forward_kl_is_nonnegative_and_prompt_mask_is_respected(self):
        _loss, metrics = core.compute_ppo_forward_kl_loss(
            [torch.tensor([10.0, -0.2], requires_grad=True)],
            old_logprobs_list=[[0.0, -0.2]],
            reference_logprobs_list=[[0.0, -1.0]],
            advantages_list=[[0.0, 1.0]],
            action_masks=[[0.0, 1.0]],
            kl_penalty_coef=0.03,
        )

        expected_kl = math.exp(-0.8) + 0.8 - 1.0
        self.assertAlmostEqual(metrics["explicit_forward_kl"], expected_kl, places=6)
        self.assertGreaterEqual(metrics["explicit_forward_kl"], 0.0)

    def test_ppo_ratio_clipping_uses_point_nine_to_one_point_one(self):
        _loss, metrics = core.compute_ppo_forward_kl_loss(
            [torch.tensor([math.log(2.0)], requires_grad=True)],
            old_logprobs_list=[[0.0]],
            reference_logprobs_list=[[math.log(2.0)]],
            advantages_list=[[1.0]],
            action_masks=[[1.0]],
            kl_penalty_coef=0.03,
        )

        self.assertAlmostEqual(metrics["ppo_loss"], -1.1, places=6)
        self.assertAlmostEqual(metrics["ppo_clip_fraction"], 1.0, places=6)

    def test_loss_rejects_sequence_length_mismatch(self):
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            core.compute_ppo_forward_kl_loss(
                [torch.tensor([-0.2, -0.3], requires_grad=True)],
                old_logprobs_list=[[-0.2]],
                reference_logprobs_list=[[-0.2, -0.3]],
                advantages_list=[[1.0, 1.0]],
                action_masks=[[1.0, 1.0]],
                kl_penalty_coef=0.03,
            )

    def test_prepare_rejects_datum_mask_length_mismatch(self):
        datum = _datum(
            target_tokens=[20, 21],
            old_logprobs=[-0.2, -0.3],
            advantages=[1.0, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            core._prepare_explicit_ppo_kl(
                [datum],
                [[1.0]],
                _ReferenceClient([0.0, -0.2, -0.3]),
                0.03,
            )

    def test_prepare_rejects_reference_logprob_length_mismatch(self):
        datum = _datum(
            target_tokens=[20, 21],
            old_logprobs=[-0.2, -0.3],
            advantages=[1.0, 1.0],
        )
        with self.assertRaisesRegex(ValueError, "reference logprob length mismatch"):
            core._prepare_explicit_ppo_kl(
                [datum],
                [[1.0, 1.0]],
                _ReferenceClient([0.0, -0.2]),
                0.03,
            )

    def test_prepare_counts_only_completion_tokens(self):
        datum = _datum(
            target_tokens=[20, 21, 22],
            old_logprobs=[0.0, -0.2, -0.3],
            advantages=[0.0, 1.0, 1.0],
        )
        custom_datums, loss_fn, kl_tokens = core._prepare_explicit_ppo_kl(
            [datum],
            [[0.0, 1.0, 1.0]],
            _ReferenceClient([0.0, 0.0, -0.2, -0.3]),
            0.03,
        )

        self.assertEqual(len(custom_datums), 1)
        self.assertEqual(kl_tokens, 2)
        weights = list(custom_datums[0].loss_fn_inputs["weights"].data)
        self.assertEqual(weights, [0.0, 1.0, 1.0])

        new_logprobs = [torch.tensor([0.0, -0.2, -0.3], requires_grad=True)]
        loss, metrics = loss_fn(None, new_logprobs)
        self.assertAlmostEqual(metrics["explicit_forward_kl"], 0.0, places=7)
        loss.backward()
        self.assertTrue(torch.all(torch.isfinite(new_logprobs[0].grad)))


if __name__ == "__main__":
    unittest.main()
