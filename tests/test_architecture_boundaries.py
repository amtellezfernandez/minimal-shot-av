from __future__ import annotations

import ast
from pathlib import Path
import unittest

from tests.pyproject_helpers import load_string_tables


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "minimal_shot_av"
MODEL_SRC = SRC / "model"
SIMULATOR_SRC = SRC / "simulator"
NEUTRAL_SRC = SRC / "neutral"
SCRIPTS = ROOT / "scripts"
PYPROJECT = ROOT / "pyproject.toml"
UV_LOCK = ROOT / "uv.lock"

MODEL_MODULES = {
    "anchor_trajectory_model",
    "kinematic_candidates",
    "learned_trajectory_model",
    "neural_trajectory_model",
    "rfs_metric",
    "trajectory_io",
    "trajectory_resampling",
    "transformer_trajectory_model",
    "wod_e2e",
    "wod_preference",
    "wod_ranker",
    "wod_submission",
    "world_model",
    "zero_shot_eval",
}

SIMULATOR_MODULES = {
    "alpasim_signal",
    "alpasim_spotlight",
    "certification",
    "compass",
    "compositional_scenarios",
    "environment",
    "oracle",
    "perception",
    "planner",
    "policy",
    "render",
    "safety",
    "spotlight_reflex",
    "trajectory_selector",
    "vehicle_command",
    "wod_scenarios",
    "world_model",
}

NEUTRAL_MODULES = {
    "__init__",
    "alpasim_metrics",
    "benchmark_compare",
    "benchmark_reports",
}

ROOT_FORBIDDEN_MODULES = MODEL_MODULES | SIMULATOR_MODULES | (NEUTRAL_MODULES - {"__init__"})

MODEL_SCRIPTS = {
    "audit_external_embedding_result.py",
    "audit_wod_validation_breakthrough.py",
    "audit_wod_improvement_target.py",
    "audit_wod_e2e_readiness.py",
    "audit_wod_model_bias.py",
    "audit_wod_reproducibility.py",
    "benchmark_wod_runtime.py",
    "build_external_embedding_cache.py",
    "build_cosmos_predict25_tokenizer_embedding_cache.py",
    "build_cosmos_tokenizer_embedding_cache.py",
    "build_wod_e2e_frame_list.py",
    "build_wod_candidate_score_dataset.py",
    "build_wod_scene_token_cache.py",
    "build_wod_preference_dataset.py",
    "check_wod_e2e_parser.py",
    "dedupe_jsonl_by_key.py",
    "evaluate_wod_zero_shot.py",
    "evaluate_wod_e2e_rfs.py",
    "evaluate_wod_preference_ranker.py",
    "evaluate_wod_anchor_model_cv.py",
    "evaluate_wod_trajectory_model_cv.py",
    "export_wod_gcs_shard_samples.py",
    "generate_wod_anchor_candidates.py",
    "generate_wod_kinematic_candidates.py",
    "generate_wod_learned_candidates.py",
    "export_wod_neural_training_frames.py",
    "train_wod_anchor_trajectory_model.py",
    "train_wod_neural_trajectory_model.py",
    "train_wod_transformer_trajectory_model.py",
    "prepare_wod_e2e_data.py",
    "prepare_wod_e2e_submission_matrix.py",
    "record_wod_leaderboard_result.py",
    "run_wod_breakthrough_experiments.py",
    "run_wod_leaderboard_attack.py",
    "score_wod_candidates_with_ranker.py",
    "stage_wod_gcs_shards.py",
    "train_wod_contextual_ranker.py",
    "train_wod_trajectory_model.py",
    "train_wod_preference_ranker.py",
    "validate_wod_e2e_submission.py",
    "write_wod_e2e_submission.py",
}

SIMULATOR_SCRIPTS = {
    "audit_novel_object_stress.py",
    "audit_alpasignal_bridge.py",
    "audit_minor_runtime_constraints.py",
    "audit_sota_submission_bundles.py",
    "build_minor_visual_gallery.py",
    "evaluate_scenarios.py",
    "run_demo.py",
    "run_submission_demos.py",
    "run_vehicle_validation_shadow.py",
}

NEUTRAL_SCRIPTS = {
    "check_gpu_acceleration.py",
    "check_cuda_preflight.py",
    "check_code_quality.py",
    "audit_hardware_vehicle_validation.py",
    "audit_sota_judging_criteria.py",
    "audit_production_av_readiness.py",
    "audit_final_submission_readiness.py",
    "audit_minimal_shot_claim.py",
    "import_alpasim_metrics.py",
    "produce_alpasim_comparable_reports.py",
    "run_tests.py",
}


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_root_package_has_no_domain_modules(self) -> None:
        self.assertEqual(["__init__"], sorted(path.stem for path in SRC.glob("*.py")))

    def test_all_model_modules_have_boundary_ownership(self) -> None:
        discovered = {path.stem for path in MODEL_SRC.glob("*.py")}
        self.assertEqual(MODEL_MODULES | {"__init__"}, discovered)

    def test_all_simulator_modules_have_boundary_ownership(self) -> None:
        discovered = {path.stem for path in SIMULATOR_SRC.glob("*.py")}
        self.assertEqual(SIMULATOR_MODULES | {"__init__"}, discovered)

    def test_all_neutral_modules_have_boundary_ownership(self) -> None:
        discovered = {path.stem for path in NEUTRAL_SRC.glob("*.py")}
        self.assertEqual(NEUTRAL_MODULES, discovered)

    def test_all_python_scripts_have_boundary_ownership(self) -> None:
        discovered = {path.name for path in SCRIPTS.glob("*.py")}
        owned = MODEL_SCRIPTS | SIMULATOR_SCRIPTS | NEUTRAL_SCRIPTS
        self.assertEqual(owned, discovered)

    def test_no_code_imports_old_flat_root_modules(self) -> None:
        paths = [
            *MODEL_SRC.glob("*.py"),
            *SIMULATOR_SRC.glob("*.py"),
            *NEUTRAL_SRC.glob("*.py"),
            *SCRIPTS.glob("*.py"),
        ]
        violations = _flat_root_import_violations(paths)
        self.assertEqual([], violations)

    def test_pyproject_entrypoints_target_boundary_packages(self) -> None:
        pyproject = load_string_tables(PYPROJECT)
        targets: list[str] = []
        targets.extend(pyproject["project.scripts"].values())
        for section, values in pyproject.items():
            if section.startswith('project.entry-points.'):
                targets.extend(values.values())

        violations = [
            target
            for target in targets
            if not target.startswith(
                (
                    "minimal_shot_av.model.",
                    "minimal_shot_av.simulator.",
                    "minimal_shot_av.neutral.",
                )
            )
        ]
        self.assertEqual([], sorted(violations))

    def test_model_modules_do_not_import_simulator_modules(self) -> None:
        violations = _domain_import_violations(MODEL_SRC.glob("*.py"), forbidden_domains={"simulator"})
        self.assertEqual([], violations)

    def test_simulator_modules_do_not_import_model_modules(self) -> None:
        violations = _domain_import_violations(SIMULATOR_SRC.glob("*.py"), forbidden_domains={"model"})
        self.assertEqual([], violations)

    def test_neutral_modules_do_not_import_model_or_simulator_modules(self) -> None:
        violations = _domain_import_violations(NEUTRAL_SRC.glob("*.py"), forbidden_domains={"model", "simulator"})
        self.assertEqual([], violations)

    def test_model_scripts_do_not_import_simulator_modules(self) -> None:
        violations = _script_boundary_violations(MODEL_SCRIPTS, forbidden_domains={"simulator"})
        self.assertEqual([], violations)

    def test_simulator_scripts_do_not_import_model_modules(self) -> None:
        violations = _script_boundary_violations(SIMULATOR_SCRIPTS, forbidden_domains={"model"})
        self.assertEqual([], violations)

    def test_neutral_scripts_do_not_import_model_or_simulator_modules(self) -> None:
        violations = _script_boundary_violations(NEUTRAL_SCRIPTS, forbidden_domains={"model", "simulator"})
        self.assertEqual([], violations)

    def test_active_code_has_no_text_prompt_model_surfaces(self) -> None:
        paths = [
            *MODEL_SRC.glob("*.py"),
            *SIMULATOR_SRC.glob("*.py"),
            *NEUTRAL_SRC.glob("*.py"),
            *SCRIPTS.glob("*.py"),
            PYPROJECT,
            UV_LOCK,
        ]
        forbidden_terms = [
            "qwen",
            "text-prompt",
            "raw_response",
            "trajectory_to_wod20_from_response",
            "write_prompt_records",
            "return json",
        ]
        violations = _text_surface_violations(paths, forbidden_terms)
        self.assertEqual([], violations)


def _domain_import_violations(paths, *, forbidden_domains: set[str]) -> list[str]:
    violations: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported in _domain_imports(tree):
            if imported in forbidden_domains:
                violations.append(f"{path.name} imports {imported}")
    return violations


def _script_boundary_violations(sources: set[str], *, forbidden_domains: set[str]) -> list[str]:
    violations: list[str] = []
    for source in sorted(sources):
        path = SCRIPTS / source
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for imported in _domain_imports(tree):
            if imported in forbidden_domains:
                violations.append(f"{source} imports {imported}")
    return violations


def _domain_imports(tree: ast.AST) -> set[str]:
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module == "minimal_shot_av":
                for alias in node.names:
                    if alias.name in {"model", "simulator", "neutral"}:
                        imports.add(alias.name)
            elif node.module and node.module.startswith("minimal_shot_av."):
                parts = node.module.split(".")
                if len(parts) >= 3:
                    imports.add(parts[1])
            elif node.level == 2 and node.module:
                imports.add(node.module.split(".", 1)[0])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("minimal_shot_av."):
                    parts = alias.name.split(".")
                    if len(parts) >= 3:
                        imports.add(parts[1])
    return imports


def _flat_root_import_violations(paths) -> list[str]:
    violations: list[str] = []
    for path in sorted(paths):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module == "minimal_shot_av":
                    for alias in node.names:
                        if alias.name in ROOT_FORBIDDEN_MODULES:
                            violations.append(f"{path.name} imports minimal_shot_av.{alias.name}")
                elif node.module and node.module.startswith("minimal_shot_av."):
                    parts = node.module.split(".")
                    if len(parts) >= 2 and parts[1] in ROOT_FORBIDDEN_MODULES:
                        violations.append(f"{path.name} imports {'.'.join(parts[:2])}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    parts = alias.name.split(".")
                    if len(parts) >= 2 and parts[0] == "minimal_shot_av" and parts[1] in ROOT_FORBIDDEN_MODULES:
                        violations.append(f"{path.name} imports {'.'.join(parts[:2])}")
    return violations


def _text_surface_violations(paths, forbidden_terms: list[str]) -> list[str]:
    violations: list[str] = []
    for path in sorted(paths):
        text = path.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            if term in text:
                violations.append(f"{path.relative_to(ROOT)} contains {term}")
    return violations


if __name__ == "__main__":
    unittest.main()
