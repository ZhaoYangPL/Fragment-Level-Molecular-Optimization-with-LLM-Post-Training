from __future__ import annotations

import importlib.util
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from rdkit import Chem


MODULE_DIR = Path(__file__).resolve().parents[1]
CORE_PATH = MODULE_DIR / "TURN2Bonus0.25KL_delta_abs_rl_core.py"
sys.path.insert(0, str(MODULE_DIR))

SPEC = importlib.util.spec_from_file_location("qed_logp_mw_training_core", CORE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Unable to load training core from {CORE_PATH}")
core = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = core
SPEC.loader.exec_module(core)


def _target_prompt(initial_smiles: str, target_smiles: str) -> str:
    initial = Chem.MolFromSmiles(initial_smiles)
    target = Chem.MolFromSmiles(target_smiles)
    if initial is None or target is None:
        raise ValueError("Test molecules must be valid")
    initial_props = core._property_values(initial)
    target_props = core._property_values(target)

    tags = []
    for tag, key in (("QED", "qed"), ("LogP", "logp"), ("MW", "mw")):
        delta = target_props[key] - initial_props[key]
        direction = "higher" if delta >= 0 else "lower"
        tags.append(f"<{tag}>{abs(delta):.12f}</{tag}> {direction}")
    return (
        f"Given molecule <SMILES>{initial_smiles}</SMILES>, make QED {tags[0]}, "
        f"LogP {tags[1]}, and Molecular Weight {tags[2]}."
    )


class _FixedRng:
    def __init__(self, indices):
        self.indices = np.asarray(indices)

    def choice(self, _size, size, replace):
        if replace:
            raise AssertionError("Prompt sampling must not use replacement")
        return self.indices[:size]


class _PromptInput:
    def to_ints(self):
        return [10, 11]


class _Renderer:
    def build_generation_prompt(self, _messages):
        return _PromptInput()

    def get_stop_sequences(self):
        return []


class _Tokenizer:
    def __init__(self, outputs):
        self.outputs = outputs

    def decode(self, tokens, skip_special_tokens=True):
        if not skip_special_tokens:
            raise AssertionError("Expected special tokens to be skipped")
        return self.outputs[int(tokens[0])]


class _CompletedFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class _SamplingClient:
    def __init__(self, token_ids):
        self.token_ids = iter(token_ids)

    def sample(self, **_kwargs):
        token_id = next(self.token_ids)
        sequence = SimpleNamespace(tokens=[token_id], logprobs=[-0.2])
        return _CompletedFuture(SimpleNamespace(sequences=[sequence]))


class TrainingLogicTests(unittest.TestCase):
    def test_property_reward_invalid_no_improvement_and_improvement(self):
        prompt = _target_prompt("CC", "CCC")

        invalid = core.compute_property_reward_details(
            "Add C to form <SMILES>not-a-smiles</SMILES>.",
            prompt,
        )
        self.assertEqual(invalid["property_reward"], -10.0)
        self.assertFalse(invalid["accepted"])

        unchanged = core.compute_property_reward_details(
            "Replace C with C to form <SMILES>CC</SMILES>.",
            prompt,
        )
        # Q-gating: Δ=0 + not improved → Q term is zeroed, property_reward = 0
        self.assertEqual(unchanged["rejection_reason"], "no_improvement")
        self.assertFalse(unchanged["accepted"])
        self.assertFalse(unchanged["improved"])

        improved = core.compute_property_reward_details(
            "Add C to form <SMILES>CCC</SMILES>.",
            prompt,
        )
        self.assertTrue(improved["improved"])
        self.assertTrue(improved["accepted"])
        self.assertGreater(improved["property_reward"], 0.0)

    def test_improvement_threshold_is_strict(self):
        self.assertFalse(core.is_nte_improvement(False, 1.0, 0.0))
        self.assertFalse(core.is_nte_improvement(True, 1.0, 1.0))
        self.assertTrue(core.is_nte_improvement(True, 1.0, 1.0 - 1e-12))

    def test_all_nte_buckets_are_eligible_for_uniform_sampling(self):
        raw_data = [
            {"input": _target_prompt("CC", "CCC"), "output": "unused"},
            {
                "input": (
                    "Given <SMILES>CC</SMILES>, make QED <QED>0</QED> higher, "
                    "LogP <LogP>3</LogP> higher, and Molecular Weight "
                    "<MW>0</MW> higher."
                ),
                "output": "unused",
            },
            {
                "input": (
                    "Given <SMILES>CC</SMILES>, make QED <QED>0</QED> higher, "
                    "LogP <LogP>9</LogP> higher, and Molecular Weight "
                    "<MW>0</MW> higher."
                ),
                "output": "unused",
            },
        ]
        buckets = [core._bucket_label_for_initial_nte(core._initial_nte_from_prompt(row["input"])) for row in raw_data]
        self.assertEqual(buckets, ["low", "mid", "high"])

        indices, stats = core._sample_prompt_indices(
            raw_data,
            _FixedRng([0, 1, 2]),
            candidate_prompts_per_iter=3,
            max_kept_prompts_per_iter=3,
        )
        self.assertEqual(indices, [0, 1, 2])
        self.assertEqual(stats["prompt_sampling_kept"], 3)

    def test_rejected_valid_molecule_is_not_next_base_or_terminal(self):
        prompt = _target_prompt("CC", "CCC")
        unchanged_output = "Replace C with C to form <SMILES>CC</SMILES>."
        tokenizer = _Tokenizer({101: unchanged_output, 102: unchanged_output})

        _datums, _masks, _returns, details = core.run_mode_b(
            {"input": prompt, "output": "unused"},
            _Renderer(),
            _SamplingClient([101, 102]),
            tokenizer,
            ema_baseline=core.EMABaseline(),
            num_trajectories=1,
            max_turns=2,
        )

        turns = details["turn_output_records"]
        self.assertEqual(len(turns), 2)
        self.assertEqual(turns[0]["rejection_reason"], "no_improvement")
        self.assertFalse(turns[0]["accepted"])
        self.assertEqual(turns[1]["input_canonical_smiles"], Chem.MolToSmiles(Chem.MolFromSmiles("CC")))

        terminal = details["terminal_molecule_records"][0]
        self.assertEqual(terminal["source"], "accepted_initial")
        self.assertEqual(terminal["canonical_smiles"], Chem.MolToSmiles(Chem.MolFromSmiles("CC")))
        self.assertIsNone(terminal["terminal_output_text"])
        self.assertEqual(terminal["last_model_output_text"], unchanged_output)

    def test_learning_rate_boundaries_and_continuation(self):
        expected = {
            0: 5e-6,
            1000: 5e-5,
            6000: 5e-5,
            25000: 1e-6,
            30000: 1e-6,
        }
        for step, value in expected.items():
            self.assertAlmostEqual(
                core.compute_scheduled_learning_rate(
                    step,
                    warmup_start_lr=5e-6,
                    max_lr=5e-5,
                    min_lr=1e-6,
                    warmup_iterations=1000,
                    hold_iterations=5000,
                    total_iterations=25000,
                ),
                value,
            )
        self.assertLess(
            core.compute_scheduled_learning_rate(
                10500,
                warmup_start_lr=5e-6,
                max_lr=5e-5,
                min_lr=1e-6,
                warmup_iterations=1000,
                hold_iterations=5000,
                total_iterations=25000,
            ),
            core.compute_scheduled_learning_rate(
                10000,
                warmup_start_lr=5e-6,
                max_lr=5e-5,
                min_lr=1e-6,
                warmup_iterations=1000,
                hold_iterations=5000,
                total_iterations=25000,
            ),
        )

    def test_explicit_forward_kl_is_zero_or_positive_and_respects_mask(self):
        zero_loss, zero_metrics = core.compute_ppo_forward_kl_loss(
            [torch.tensor([-0.2], requires_grad=True)],
            old_logprobs_list=[[-0.2]],
            reference_logprobs_list=[[-0.2]],
            advantages_list=[[1.0]],
            action_masks=[[1.0]],
            kl_penalty_coef=0.05,
        )
        self.assertAlmostEqual(zero_metrics["explicit_forward_kl"], 0.0, places=7)
        zero_loss.backward()

        _loss, metrics = core.compute_ppo_forward_kl_loss(
            [torch.tensor([10.0, -0.2], requires_grad=True)],
            old_logprobs_list=[[0.0, -0.2]],
            reference_logprobs_list=[[0.0, -1.0]],
            advantages_list=[[0.0, 1.0]],
            action_masks=[[0.0, 1.0]],
            kl_penalty_coef=0.05,
        )
        expected_kl = math.exp(-0.8) + 0.8 - 1.0
        self.assertAlmostEqual(metrics["explicit_forward_kl"], expected_kl, places=6)
        self.assertGreater(metrics["explicit_forward_kl"], 0.0)

    def test_ppo_ratio_clipping_is_preserved(self):
        _loss, metrics = core.compute_ppo_forward_kl_loss(
            [torch.tensor([math.log(2.0)], requires_grad=True)],
            old_logprobs_list=[[0.0]],
            reference_logprobs_list=[[math.log(2.0)]],
            advantages_list=[[1.0]],
            action_masks=[[1.0]],
            kl_penalty_coef=0.05,
        )
        self.assertAlmostEqual(metrics["ppo_loss"], -1.2, places=6)
        self.assertAlmostEqual(metrics["ppo_clip_fraction"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
