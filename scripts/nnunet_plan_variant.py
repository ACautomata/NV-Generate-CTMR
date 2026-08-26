#!/usr/bin/env python3
"""Derive the audited SSA 8-card batch-16 nnU-Net plan variant.

This tool never changes the source plan. It adds only the approved
``3d_fullres_bs16`` configuration to a new plans file and writes a compact
sidecar audit record for the controlled data directory.
"""

import argparse
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

PLANS_IDENTIFIER = "nnUNetPlans_SSA_bs16_v1"
PARENT_CONFIGURATION = "3d_fullres"
VARIANT_CONFIGURATION = "3d_fullres_bs16"
GLOBAL_BATCH_SIZE = 16
WORLD_SIZE = 8
PERSISTENT_ROOT = Path("/root/private_data")


class PlanVariantBuilder:
    """Builds and verifies the approved SSA batch-16 plans variant."""

    def __init__(self, source_plans: Path, derived_plans: Path, audit_path: Path):
        self.source_plans = source_plans
        self.derived_plans = derived_plans
        self.audit_path = audit_path

    def build(self) -> dict:
        self._reject_overwrite()
        source_bytes = self.source_plans.read_bytes()
        source = json.loads(source_bytes)
        self._validate_source(source)

        derived = copy.deepcopy(source)
        derived["plans_name"] = PLANS_IDENTIFIER
        derived["configurations"][VARIANT_CONFIGURATION] = {
            "inherits_from": PARENT_CONFIGURATION,
            "batch_size": GLOBAL_BATCH_SIZE,
        }

        parent = self._resolve_configuration(source["configurations"], PARENT_CONFIGURATION)
        variant = self._resolve_configuration(derived["configurations"], VARIANT_CONFIGURATION)
        self._assert_approved_delta(parent, variant)

        derived_bytes = (json.dumps(derived, indent=2, sort_keys=True) + "\n").encode()
        audit = self._build_audit(source_bytes, derived_bytes, parent, variant)
        self.derived_plans.write_bytes(derived_bytes)
        self.audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        return audit

    def _reject_overwrite(self) -> None:
        for label, path in {
            "source_plans": self.source_plans,
            "derived_plans": self.derived_plans,
            "audit_path": self.audit_path,
        }.items():
            if not path.resolve().is_relative_to(PERSISTENT_ROOT):
                raise ValueError(f"{label} must be under {PERSISTENT_ROOT}: {path}")
        if not self.source_plans.is_file():
            raise FileNotFoundError(f"source plans not found: {self.source_plans}")
        if self.source_plans.resolve() == self.derived_plans.resolve():
            raise ValueError("derived plans must not overwrite the source plans")
        if self.derived_plans.exists():
            raise FileExistsError(f"refusing to overwrite derived plans: {self.derived_plans}")
        if self.audit_path.exists():
            raise FileExistsError(f"refusing to overwrite audit record: {self.audit_path}")
        self.derived_plans.parent.mkdir(parents=True, exist_ok=True)
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    def _validate_source(self, source: dict) -> None:
        if source.get("plans_name") != "nnUNetPlans":
            raise ValueError(f"expected source plans_name=nnUNetPlans, got {source.get('plans_name')!r}")
        configurations = source.get("configurations")
        if not isinstance(configurations, dict) or PARENT_CONFIGURATION not in configurations:
            raise ValueError(f"source plans lack {PARENT_CONFIGURATION!r} configuration")
        if VARIANT_CONFIGURATION in configurations:
            raise ValueError(f"source plans already contain {VARIANT_CONFIGURATION!r}")

    def _resolve_configuration(self, configurations: dict, name: str, seen: tuple[str, ...] = ()) -> dict:
        if name in seen:
            raise ValueError(f"cyclic configuration inheritance: {' -> '.join((*seen, name))}")
        configuration = configurations.get(name)
        if not isinstance(configuration, dict):
            raise ValueError(f"configuration not found: {name!r}")
        parent_name = configuration.get("inherits_from")
        resolved = {}
        if parent_name is not None:
            resolved.update(self._resolve_configuration(configurations, parent_name, (*seen, name)))
        resolved.update({key: value for key, value in configuration.items() if key != "inherits_from"})
        return resolved

    def _assert_approved_delta(self, parent: dict, variant: dict) -> None:
        parent_fields = self._flatten(parent)
        variant_fields = self._flatten(variant)
        changed = {
            field: {"parent": parent_fields.get(field), "variant": variant_fields.get(field)}
            for field in sorted(set(parent_fields) | set(variant_fields))
            if parent_fields.get(field) != variant_fields.get(field)
        }
        if changed != {"batch_size": {"parent": parent.get("batch_size"), "variant": GLOBAL_BATCH_SIZE}}:
            raise ValueError(f"derived configuration changes fields beyond batch_size: {changed}")
        if parent.get("batch_size") != 2:
            raise ValueError(f"expected SSA parent global batch_size=2, got {parent.get('batch_size')!r}")
        if variant.get("batch_size") != GLOBAL_BATCH_SIZE:
            raise ValueError(f"expected derived global batch_size={GLOBAL_BATCH_SIZE}")
        if variant.get("data_identifier") != parent.get("data_identifier"):
            raise ValueError("batch-only variant must reuse the parent's data_identifier")
        if GLOBAL_BATCH_SIZE % WORLD_SIZE:
            raise ValueError("global batch size must be divisible by the approved world size")

    def _flatten(self, value: dict, prefix: str = "") -> dict:
        flattened = {}
        for key, item in value.items():
            field = f"{prefix}.{key}" if prefix else key
            if isinstance(item, dict):
                flattened.update(self._flatten(item, field))
            else:
                flattened[field] = item
        return flattened

    def _build_audit(self, source_bytes: bytes, derived_bytes: bytes, parent: dict, variant: dict) -> dict:
        return {
            "schema_version": 1,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dataset": "Dataset502_BraTS2023SSA",
            "scope": "SSA fold_0 8-card derived-plan instrument only",
            "plans_identifier": PLANS_IDENTIFIER,
            "parent_configuration": PARENT_CONFIGURATION,
            "variant_configuration": VARIANT_CONFIGURATION,
            "source_plans_sha256": hashlib.sha256(source_bytes).hexdigest(),
            "derived_plans_sha256": hashlib.sha256(derived_bytes).hexdigest(),
            "source_global_batch_size": parent["batch_size"],
            "derived_global_batch_size": variant["batch_size"],
            "world_size": WORLD_SIZE,
            "local_batch_size": variant["batch_size"] // WORLD_SIZE,
            "data_identifier": variant["data_identifier"],
            "approved_delta": {"batch_size": GLOBAL_BATCH_SIZE},
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-plans", type=Path, required=True)
    parser.add_argument("--derived-plans", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()

    audit = PlanVariantBuilder(args.source_plans, args.derived_plans, args.audit).build()
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
