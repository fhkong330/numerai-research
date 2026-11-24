"""Package and validate the frozen V1 function for Numerai Model Uploads."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
MODEL_VERSION = "v6_v1_production_002"
HOSTED_VERSION = "v6_v1_hosted_001"
PACKAGE_FILENAME = "numerai_v6_v1_production_002.pkl"
OFFICIAL_REVISION = "e782c7dc92f0e7aaf41219f90d3a9d768df6c84b"
OFFICIAL_REQUIREMENTS_URL = (
    "https://github.com/numerai/numerai-predict/blob/master/py3.12/requirements.txt"
)
OFFICIAL_DOCS_URL = "https://docs.numer.ai/numerai-tournament/submissions/model-uploads"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feature_list_sha256(features) -> str:
    return hashlib.sha256("\n".join(features).encode("utf-8")).hexdigest()


def write_json_once(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, allow_nan=False)


def verify_source_model_artifacts(model_dir: Path) -> dict:
    marker_path = model_dir / "PRODUCTION_TRAINING_COMPLETED.json"
    if not marker_path.is_file():
        raise ValueError("Production model artifacts are incomplete.")
    marker = json.loads(marker_path.read_text())
    for name, digest in marker["artifact_sha256"].items():
        if sha256_file(model_dir / name) != digest:
            raise ValueError(f"Production model artifact hash changed: {name}")
    config = json.loads((model_dir / "production_config.json").read_text())
    for name in ("medium.txt", "top250.txt"):
        if lgb.Booster(model_file=str(model_dir / "models" / name)).num_trees() != 300:
            raise ValueError(f"Unexpected production model tree count: {name}")
    return {"config": config, "completion": marker}


def verify_reference_artifacts(reference_dir: Path) -> dict:
    marker_path = reference_dir / "LIVE_INFERENCE_COMPLETED.json"
    if not marker_path.is_file():
        raise ValueError("Live reference artifacts are incomplete.")
    marker = json.loads(marker_path.read_text())
    for name, digest in marker["artifact_sha256"].items():
        if sha256_file(reference_dir / name) != digest:
            raise ValueError(f"Live reference artifact hash changed: {name}")
    return marker


def make_hosted_predict(
    medium_model_text: str,
    top250_model_text: str,
    medium_features: tuple[str, ...],
    top250_features: tuple[str, ...],
):
    """Return the self-contained official two-argument prediction closure."""
    medium_features = tuple(medium_features)
    top250_features = tuple(top250_features)

    def predict(live_features, live_benchmark_models):
        import lightgbm as hosted_lgb
        import numpy as hosted_np
        import pandas as hosted_pd

        del live_benchmark_models
        if not live_features.index.is_unique or live_features.index.hasnans:
            raise ValueError("Hosted live input index must be present and unique.")
        missing = [name for name in medium_features if name not in live_features.columns]
        if missing:
            raise ValueError(f"Hosted live input is missing required features: {missing[:5]}")
        for name in medium_features:
            if not hosted_pd.api.types.is_numeric_dtype(live_features[name]):
                raise ValueError(f"Hosted feature must be numeric: {name}")
        medium_model = hosted_lgb.Booster(model_str=medium_model_text)
        top250_model = hosted_lgb.Booster(model_str=top250_model_text)
        medium_raw = hosted_np.asarray(
            medium_model.predict(
                live_features.loc[:, list(medium_features)], num_threads=1
            ),
            dtype=hosted_np.float64,
        )
        top250_raw = hosted_np.asarray(
            top250_model.predict(
                live_features.loc[:, list(top250_features)], num_threads=1
            ),
            dtype=hosted_np.float64,
        )
        if not hosted_np.isfinite(medium_raw).all() or not hosted_np.isfinite(top250_raw).all():
            raise ValueError("Hosted component predictions must be finite.")
        count = len(live_features)
        medium_rank = (
            hosted_pd.Series(medium_raw, index=live_features.index).rank(method="average") - 0.5
        ) / count
        top250_rank = (
            hosted_pd.Series(top250_raw, index=live_features.index).rank(method="average") - 0.5
        ) / count
        ensemble = 0.5 * medium_rank + 0.5 * top250_rank
        if not hosted_np.isfinite(ensemble.to_numpy(dtype=hosted_np.float64)).all():
            raise ValueError("Hosted ensemble predictions must be finite.")
        if not ensemble.between(0.0, 1.0, inclusive="both").all():
            raise ValueError("Hosted ensemble predictions must be within [0, 1].")
        return ensemble.to_frame("prediction")

    return predict


def _versions() -> dict:
    result = {"python": sys.version.split()[0]}
    for package in ("cloudpickle", "lightgbm", "numpy", "pandas", "pyarrow", "scipy"):
        result[package] = importlib.metadata.version(package)
    return result


def _load_source_identity(model_dir: Path) -> dict:
    verified = verify_source_model_artifacts(model_dir)
    if verified["config"]["model_version"] != MODEL_VERSION:
        raise ValueError("Hosted package requires the approved production model version.")
    source = json.loads((model_dir / "source_snapshot_manifest.json").read_text())
    selected = json.loads((model_dir / "selected_features.json").read_text())
    hashes = json.loads((model_dir / "artifact_hashes.json").read_text())
    medium = tuple(source["local_audit"]["schema"]["medium_features"])
    top250 = tuple(selected["model_feature_names"])
    if len(medium) != 350 or len(top250) != 250 or not set(top250).issubset(medium):
        raise ValueError("Hosted package feature identity is invalid.")
    return {
        "config": verified["config"],
        "medium_features": medium,
        "top250_features": top250,
        "artifact_hashes": hashes,
        "selected_manifest_sha256": sha256_file(model_dir / "selected_features.json"),
    }


def _rank(raw: np.ndarray, index: pd.Index) -> np.ndarray:
    return ((pd.Series(raw, index=index).rank(method="average") - 0.5) / len(index)).to_numpy()


def _max_difference(left, right) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def build_and_validate_package(
    model_dir: Path,
    live_path: Path,
    reference_dir: Path,
    destination: Path,
    *,
    runner_path: Path,
) -> Path:
    if destination.exists():
        raise FileExistsError(f"Hosted package version already exists: {destination}")
    destination.mkdir(parents=True)
    started = time.perf_counter()
    try:
        import cloudpickle

        identity = _load_source_identity(model_dir)
        medium_text = (model_dir / "models" / "medium.txt").read_text(encoding="utf-8")
        top250_text = (model_dir / "models" / "top250.txt").read_text(encoding="utf-8")
        medium_features = identity["medium_features"]
        top250_features = identity["top250_features"]
        live = pd.read_parquet(live_path, columns=["numerai_ticker", *medium_features]).set_index(
            "numerai_ticker"
        )
        medium_reference = pd.read_parquet(reference_dir / "medium_predictions.parquet").set_index(
            "numerai_ticker"
        ).loc[live.index]
        top250_reference = pd.read_parquet(reference_dir / "top250_predictions.parquet").set_index(
            "numerai_ticker"
        ).loc[live.index]
        ensemble_reference = pd.read_parquet(reference_dir / "ensemble_predictions.parquet").set_index(
            "numerai_ticker"
        ).loc[live.index]
        medium_model = lgb.Booster(model_str=medium_text)
        top250_model = lgb.Booster(model_str=top250_text)
        medium_raw = medium_model.predict(live.loc[:, list(medium_features)], num_threads=1)
        top250_raw = top250_model.predict(live.loc[:, list(top250_features)], num_threads=1)
        medium_rank = _rank(medium_raw, live.index)
        top250_rank = _rank(top250_raw, live.index)
        ensemble = 0.5 * medium_rank + 0.5 * top250_rank
        cross_version = {
            "medium_raw_max_abs_difference": _max_difference(
                medium_raw, medium_reference["raw_prediction"]
            ),
            "top250_raw_max_abs_difference": _max_difference(
                top250_raw, top250_reference["raw_prediction"]
            ),
            "medium_rank_max_abs_difference": _max_difference(
                medium_rank, medium_reference["prediction"]
            ),
            "top250_rank_max_abs_difference": _max_difference(
                top250_rank, top250_reference["prediction"]
            ),
            "ensemble_max_abs_difference": _max_difference(
                ensemble, ensemble_reference["prediction"]
            ),
            "medium_rank_changed_rows": int(
                np.count_nonzero(medium_rank != medium_reference["prediction"].to_numpy())
            ),
            "top250_rank_changed_rows": int(
                np.count_nonzero(top250_rank != top250_reference["prediction"].to_numpy())
            ),
            "ensemble_tolerance": 1e-12,
        }
        if cross_version["ensemble_max_abs_difference"] > 1e-12:
            raise ValueError("Official-version ensemble differs materially from saved reference.")
        hosted_predict = make_hosted_predict(
            medium_text, top250_text, medium_features, top250_features
        )
        package_path = destination / PACKAGE_FILENAME
        with package_path.open("xb") as package_file:
            cloudpickle.dump(hosted_predict, package_file)
        metadata = {
            "hosted_version": HOSTED_VERSION,
            "package_filename": PACKAGE_FILENAME,
            "source_model_version": MODEL_VERSION,
            "source_model_completion_sha256": sha256_file(
                model_dir / "PRODUCTION_TRAINING_COMPLETED.json"
            ),
            "source_model_hashes": {
                "medium": identity["artifact_hashes"]["models/medium.txt"],
                "top250": identity["artifact_hashes"]["models/top250.txt"],
            },
            "medium_feature_list_sha256": feature_list_sha256(medium_features),
            "top250_feature_list_sha256": feature_list_sha256(top250_features),
            "selected_manifest_sha256": identity["selected_manifest_sha256"],
            "frozen_policy_id": identity["config"]["frozen_policy_id"],
            "frozen_policy_sha256": identity["config"]["frozen_policy_sha256"],
            "weights": {"medium": 0.5, "top250": 0.5},
            "normalization": "average rank normalized as (rank - 0.5) / n",
            "official_contract": {
                "verified_date": "2026-11-24",
                "docs_url": OFFICIAL_DOCS_URL,
                "requirements_url": OFFICIAL_REQUIREMENTS_URL,
                "numerai_predict_revision": OFFICIAL_REVISION,
                "function_arguments": ["live_features", "live_benchmark_models"],
                "output_columns": ["prediction"],
                "input_index_preserved": True,
                "feature_country_excluded": True,
                "cpu": 1,
                "memory_gb": 4,
                "runtime_minutes": 10,
                "internet": False,
                "published_package_size_limit": None,
            },
            "environment": _versions(),
        }
        write_json_once(destination / "package_metadata.json", metadata)
        write_json_once(destination / "cross_version_compatibility.json", cross_version)

        runner_results = []
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            for label, shuffled in (("ordered", False), ("shuffled", True)):
                output_path = temporary_path / f"{label}.parquet"
                report_path = temporary_path / f"{label}.json"
                command = [
                    sys.executable,
                    str(runner_path.resolve()),
                    "--package", str(package_path.resolve()),
                    "--live", str(live_path.resolve()),
                    "--output", str(output_path),
                    "--report", str(report_path),
                ]
                if shuffled:
                    command.append("--shuffle")
                environment = os.environ.copy()
                environment["PYTHONPATH"] = ""
                for name in list(environment):
                    if name.startswith("NUMERAI_"):
                        del environment[name]
                subprocess.run(
                    command,
                    cwd=temporary_path,
                    env=environment,
                    check=True,
                    capture_output=True,
                    text=True,
                )
                predicted = pd.read_parquet(output_path).set_index("numerai_ticker").loc[
                    ensemble_reference.index
                ]
                difference = _max_difference(
                    predicted["prediction"], ensemble_reference["prediction"]
                )
                if difference > 1e-12:
                    raise ValueError(f"Standalone {label} predictions differ from reference.")
                report = json.loads(report_path.read_text())
                report["reference_max_abs_difference"] = difference
                runner_results.append({"case": label, **report})
            missing = subprocess.run(
                [
                    sys.executable, str(runner_path.resolve()),
                    "--package", str(package_path.resolve()),
                    "--live", str(live_path.resolve()),
                    "--output", str(temporary_path / "missing.parquet"),
                    "--report", str(temporary_path / "missing.json"),
                    "--drop-feature", medium_features[0],
                ],
                cwd=temporary_path,
                env=environment,
                capture_output=True,
                text=True,
            )
            if missing.returncode == 0 or "missing required features" not in missing.stderr:
                raise ValueError("Standalone package did not reject a missing required feature.")
        standalone = {
            "cases": runner_results,
            "missing_feature_rejected": True,
            "isolated_working_directory": True,
            "pythonpath_cleared": True,
            "credentials_removed": True,
            "official_container_executed": False,
            "container_limitation": "Docker was not installed locally.",
        }
        write_json_once(destination / "standalone_validation.json", standalone)
        runtime = {"total_build_and_validation_seconds": time.perf_counter() - started}
        write_json_once(destination / "runtime.json", runtime)
        package_identity = {
            "size_bytes": package_path.stat().st_size,
            "sha256": sha256_file(package_path),
        }
        write_json_once(destination / "package_identity.json", package_identity)
        names = [
            PACKAGE_FILENAME,
            "package_metadata.json",
            "cross_version_compatibility.json",
            "standalone_validation.json",
            "runtime.json",
            "package_identity.json",
        ]
        hashes = {name: sha256_file(destination / name) for name in names}
        write_json_once(destination / "artifact_hashes.json", hashes)
        write_json_once(
            destination / "HOSTED_PACKAGE_COMPLETED.json",
            {"state": "complete", "artifact_sha256": hashes},
        )
        verify_hosted_package(destination)
        return destination
    except BaseException as error:
        write_json_once(
            destination / "HOSTED_PACKAGE_FAILED.json",
            {"error_type": type(error).__name__, "error": str(error)},
        )
        raise


def verify_hosted_package(destination: Path) -> dict:
    marker_path = destination / "HOSTED_PACKAGE_COMPLETED.json"
    if not marker_path.is_file():
        raise ValueError("Hosted package is incomplete.")
    marker = json.loads(marker_path.read_text())
    for name, digest in marker["artifact_sha256"].items():
        if sha256_file(destination / name) != digest:
            raise ValueError(f"Hosted artifact hash changed: {name}")
    return marker


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hosted-version", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--live-snapshot", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--confirm-hosted-package", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.run or not args.confirm_hosted_package:
        raise SystemExit("Hosted packaging requires explicit execution confirmation.")
    if args.hosted_version != HOSTED_VERSION or args.model_version != MODEL_VERSION:
        raise SystemExit("Only the approved hosted and source model versions are supported.")
    model_dir = OUTPUT_ROOT / "v6.0" / "production" / "models" / args.model_version
    verify_reference_artifacts(args.reference_dir)
    destination = (
        OUTPUT_ROOT / "v6.0" / "production" / "hosted_models" / args.hosted_version
    )
    print(
        build_and_validate_package(
            model_dir,
            args.live_snapshot / "live.parquet",
            args.reference_dir,
            destination,
            runner_path=PROJECT_ROOT / "src" / "hosted_validation_runner.py",
        )
    )


if __name__ == "__main__":
    main()
