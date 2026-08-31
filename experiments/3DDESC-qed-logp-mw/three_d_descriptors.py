from __future__ import annotations

import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors3D
from rdkit.Chem import rdFreeSASA


RDLogger.DisableLog("rdApp.*")

# 6 descriptors, 3 groups — each group maps naturally to one 2D target
DESCRIPTOR_GROUPS = {
    "size": {
        "label": "Molecular size",
        "keys": ("volume", "sasa"),
        "descriptions": {
            "volume": "van der Waals volume (Å³), related to molecular weight",
            "sasa": "solvent-accessible surface area (Å²)",
        },
    },
    "shape": {
        "label": "Molecular shape",
        "keys": ("npr1", "npr2"),
        "descriptions": {
            "npr1": "normalized principal moment ratio I1/I3 (1=sphere, 0=rod-like)",
            "npr2": "normalized principal moment ratio I2/I3 (1=sphere, 0=disc-like)",
        },
    },
    "surface": {
        "label": "Surface chemistry",
        "keys": ("hydrophobic_fraction", "polar_charge_separation"),
        "descriptions": {
            "hydrophobic_fraction": "proportion of SASA from hydrophobic atoms (C/F/Cl/Br/I plus C–H bonds)",
            "polar_charge_separation": "spatial separation of Gasteiger partial charges (Debye), related to polarity",
        },
    },
}

ALL_DESCRIPTOR_KEYS = tuple(
    key for group in DESCRIPTOR_GROUPS.values() for key in group["keys"]
)
DESCRIPTOR_QUANTILES = (0.33, 0.66)
SCHEMA_VERSION = 2

_HYDROPHOBIC_ATOMIC_NUMS = {6, 9, 17, 35, 53}


def default_quantile_cache_path(
    output_root: str | Path,
    *,
    sample_size: int = 20000,
    seed: int = 42,
) -> Path:
    sample_part = "all" if int(sample_size) <= 0 else f"sample{int(sample_size)}"
    return (
        Path(output_root)
        / "out"
        / "3d_quantiles"
        / f"qed_logp_mw_3d_quantiles_v2_{sample_part}_seed{int(seed)}.json"
    )


def extract_first_smiles(text: str | None) -> str | None:
    if not text:
        return None
    match = re.search(r"<SMILES>(.*?)</SMILES>", str(text), flags=re.DOTALL)
    return match.group(1).strip() if match else None


def _to_float(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("non-finite descriptor value")
    return out


def _embed_molecule(mol: Chem.Mol, *, seed: int) -> int:
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    try:
        conf_id = AllChem.EmbedMolecule(mol, params, maxAttempts=50)
    except TypeError:
        conf_id = AllChem.EmbedMolecule(mol, params)

    if conf_id != -1:
        return int(conf_id)

    params.useRandomCoords = True
    try:
        return int(AllChem.EmbedMolecule(mol, params, maxAttempts=50))
    except TypeError:
        return int(AllChem.EmbedMolecule(mol, params))


def _optimize_molecule(mol: Chem.Mol) -> str:
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
        if mmff_props is not None:
            try:
                status = AllChem.MMFFOptimizeMolecule(mol, mmff_props, maxIters=200)
            except TypeError:
                status = AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
            return "mmff_ok" if int(status) == 0 else "mmff_not_converged"
        mmff_status = "mmff_unavailable"
    except Exception:
        mmff_status = "mmff_failed"

    if mmff_status in {"mmff_unavailable", "mmff_failed"}:
        try:
            status = AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            return "uff_ok" if int(status) == 0 else "uff_not_converged"
        except Exception:
            return "uff_failed"
    return mmff_status


def _is_hydrophobic_surface_atom(atom: Chem.Atom) -> bool:
    atomic_num = atom.GetAtomicNum()
    if atomic_num in _HYDROPHOBIC_ATOMIC_NUMS:
        return True
    if atomic_num == 1:
        return any(neighbor.GetAtomicNum() == 6 for neighbor in atom.GetNeighbors())
    return False


def calculate_3d_descriptors(mol: Chem.Mol, *, conf_id: int) -> dict[str, float]:
    volume = _to_float(AllChem.ComputeMolVolume(mol, confId=int(conf_id)))
    radii = rdFreeSASA.classifyAtoms(mol)
    sasa = _to_float(rdFreeSASA.CalcSASA(mol, radii, confIdx=int(conf_id)))
    if sasa <= 0.0:
        raise ValueError("non-positive SASA")

    hydrophobic_sasa = 0.0
    for atom in mol.GetAtoms():
        if not _is_hydrophobic_surface_atom(atom):
            continue
        hydrophobic_sasa += float(atom.GetProp("SASA")) if atom.HasProp("SASA") else 0.0

    AllChem.ComputeGasteigerCharges(mol)
    conf = mol.GetConformer(int(conf_id))
    charge_vector = np.zeros(3, dtype=np.float64)
    for atom in mol.GetAtoms():
        raw_charge = atom.GetProp("_GasteigerCharge") if atom.HasProp("_GasteigerCharge") else "0.0"
        try:
            charge = float(raw_charge)
        except ValueError:
            charge = 0.0
        if not math.isfinite(charge):
            charge = 0.0
        pos = conf.GetAtomPosition(atom.GetIdx())
        charge_vector += charge * np.asarray([pos.x, pos.y, pos.z], dtype=np.float64)

    npr1 = _to_float(Descriptors3D.NPR1(mol, confId=int(conf_id)))
    npr2 = _to_float(Descriptors3D.NPR2(mol, confId=int(conf_id)))

    return {
        "volume": volume,
        "sasa": sasa,
        "npr1": npr1,
        "npr2": npr2,
        "hydrophobic_fraction": _to_float(hydrophobic_sasa / sasa),
        "polar_charge_separation": _to_float(np.linalg.norm(charge_vector)),
    }


def compute_3d_descriptors_safe(
    smiles: str | None,
    *,
    seed: int = 42,
) -> tuple[dict[str, float] | None, dict[str, Any]]:
    status: dict[str, Any] = {
        "status": None,
        "valid_smiles": False,
        "embed_success": False,
        "descriptor_success": False,
    }
    if not smiles:
        status["status"] = "missing_smiles"
        return None, status

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        status["status"] = "invalid_smiles"
        return None, status

    status["valid_smiles"] = True
    mol = Chem.AddHs(mol)
    conf_id = _embed_molecule(mol, seed=seed)
    if conf_id == -1:
        status["status"] = "embed_failed"
        return None, status

    status["embed_success"] = True
    opt_status = _optimize_molecule(mol)
    status["status"] = opt_status
    if opt_status == "uff_failed":
        return None, status

    try:
        descriptors = calculate_3d_descriptors(mol, conf_id=conf_id)
    except Exception as exc:
        status["status"] = "descriptor_failed"
        status["failure_detail"] = str(exc)
        return None, status

    status["descriptor_success"] = True
    return descriptors, status


def _get_example_value(example: Any, key: str) -> Any:
    if hasattr(example, "get"):
        return example.get(key)
    return example[key]


def _sample_indices(raw_data: Any, *, sample_size: int, seed: int) -> list[int]:
    data_len = len(raw_data)
    if data_len <= 0:
        return []
    if sample_size <= 0:
        return list(range(int(data_len)))
    count = min(int(sample_size), int(data_len))
    rng = np.random.default_rng(int(seed))
    return [int(value) for value in rng.choice(data_len, size=count, replace=False)]


def _quantiles_for_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"q33": None, "q66": None, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    q33, q66 = np.quantile(arr, DESCRIPTOR_QUANTILES)
    return {"q33": float(q33), "q66": float(q66), "count": int(arr.size)}


def build_3d_quantile_cache(
    raw_data: Any,
    *,
    prompt_col: str,
    cache_path: str | Path,
    sample_size: int = 20000,
    seed: int = 42,
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    indices = _sample_indices(raw_data, sample_size=sample_size, seed=seed)
    descriptor_values = {key: [] for key in ALL_DESCRIPTOR_KEYS}
    status_counts: Counter[str] = Counter()
    records = []

    for idx in indices:
        example = raw_data[idx]
        prompt_text = str(_get_example_value(example, prompt_col))
        smiles = extract_first_smiles(prompt_text)
        descriptors, status = compute_3d_descriptors_safe(smiles, seed=seed)
        status_name = str(status.get("status") or "unknown")
        status_counts[status_name] += 1
        record = {"index": int(idx), "status": status_name, "smiles": smiles}
        if descriptors is not None:
            for key in ALL_DESCRIPTOR_KEYS:
                descriptor_values[key].append(float(descriptors[key]))
            record["descriptors"] = descriptors
        records.append(record)

    quantiles = {
        key: _quantiles_for_values(values)
        for key, values in descriptor_values.items()
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "sample_size_requested": int(sample_size),
        "sample_size_actual": int(len(indices)),
        "seed": int(seed),
        "prompt_col": prompt_col,
        "descriptor_keys": list(ALL_DESCRIPTOR_KEYS),
        "quantiles": quantiles,
        "status_counts": dict(status_counts),
        "records": records,
        "cache_hit": False,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_or_build_3d_quantile_cache(
    raw_data: Any,
    *,
    prompt_col: str,
    cache_path: str | Path,
    sample_size: int = 20000,
    seed: int = 42,
    refresh: bool = False,
) -> dict[str, Any]:
    cache_path = Path(cache_path)
    if cache_path.exists() and not refresh:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        payload["cache_hit"] = True
        return payload
    return build_3d_quantile_cache(
        raw_data, prompt_col=prompt_col, cache_path=cache_path,
        sample_size=sample_size, seed=seed,
    )


def format_3d_context(
    descriptors: dict[str, float],
    *,
    enabled_groups: set[str] | None = None,
) -> str | None:
    """Build a concise 3D descriptor summary for prompt injection.

    Only includes groups whose names appear in *enabled_groups*.
    Pure data — no labels, no suggestions, no quantile annotations.
    """
    if enabled_groups is None:
        enabled_groups = set(DESCRIPTOR_GROUPS)

    lines = []
    for group_name, group_info in DESCRIPTOR_GROUPS.items():
        if group_name not in enabled_groups:
            continue
        parts = []
        for key in group_info["keys"]:
            if key not in descriptors:
                continue
            desc = group_info["descriptions"].get(key, key)
            parts.append(f"{key}={float(descriptors[key]):.3f} ({desc})")
        if parts:
            lines.append(f"{group_info['label']}:")
            for p in parts:
                lines.append(f"  - {p}")

    return "\n".join(lines) if lines else None


def build_3d_context_for_smiles(
    smiles: str | None,
    *,
    seed: int = 42,
    enabled_groups: set[str] | None = None,
) -> dict[str, Any]:
    if enabled_groups is None or not enabled_groups:
        return {
            "enabled": True,
            "injected": False,
            "status": "no_groups_enabled",
            "context": None,
            "descriptors": None,
        }

    descriptors, status = compute_3d_descriptors_safe(smiles, seed=seed)
    if descriptors is None:
        return {
            "enabled": True,
            "injected": False,
            "status": status.get("status"),
            "context": None,
            "descriptors": None,
        }

    return {
        "enabled": True,
        "injected": True,
        "status": status.get("status"),
        "context": format_3d_context(descriptors, enabled_groups=enabled_groups),
        "descriptors": descriptors,
    }
