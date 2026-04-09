import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import mlflow
from google.cloud import storage

logger = logging.getLogger(__name__)


def __upsert_model_artifacts(model_source: str, version: str, env: str):
    """Download model artifacts from MLflow and upload to GCS.

    The GCS path is model/<env>/<version>/, e.g. model/champion/1/ for champion model version 1.

    Parameters
    ----------
    model_source: str
        MLflow artifact URI for the model version, e.g. "models:/BankingModel-prod/1"
    version: str
        Model version number, used for logging and GCS path
    env: str
        Environment name, e.g. "champion" or "challenger", used for GCS path
    """

    # Download model artifacts from MLflow to a temp directory
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=model_source,
            dst_path=tmp_dir,
        )
        logger.info(
            f"Model source {model_source} (v{version}) downloaded to {local_path}"
        )

        # Upload all files to GCS under model/<env>/<version>/
        bucket = storage.Client().bucket("banking_model_assets")
        local_root = Path(local_path)
        for local_file in local_root.rglob("*"):
            if local_file.is_file():
                relative = local_file.relative_to(local_root)
                blob_path = f"model/{env}/{version}/{relative}"
                blob = bucket.blob(blob_path)
                blob.upload_from_filename(str(local_file))

    logger.info(f"Done: model/{env}/{version}/ uploaded to GCS.")


def __upload_manifest(manifest_dict: dict[str, Any]):
    """ " Upload model manifest dict to GCS as JSON file.

    Parameters
    ----------
    manifest_dict: dict[str, Any]
        Manifest data to upload, expected to contain at least champion and challenger model info with version numbers, e.g.
    """
    try:
        # Save manifest dict to a temp file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as tmp:
            json.dump(manifest_dict, tmp, indent=4)
            tmp_path = tmp.name

        # Upload temp file to GCS
        bucket = storage.Client().bucket("banking_model_assets")
        blob = bucket.blob("manifest.json")
        blob.upload_from_filename(tmp_path)

        # Clean up temp file
        os.remove(tmp_path)

    except Exception as e:
        logger.error(f"Error uploading manifest to GCS: {e}")
        raise


def __mlflow_version_to_dict(model_version) -> dict[str, Any]:
    """Convert MLflow ModelVersion object to a dict with expected fields for our API."""
    return {
        "name": model_version.name,
        "aliases": list(model_version.aliases),
        "version": model_version.version,
        "source": model_version.source,
        "run_id": model_version.run_id,
        "commit_sha": model_version.tags.get("commit.sha", "unknown"),
        "commit_head_sha": model_version.tags.get("commit.head.sha", "unknown"),
        "model_features": model_version.tags.get("model.features", "unknown"),
        "model_architecture": model_version.tags.get("model.architecture", "unknown"),
    }


def __get_active_model_version(model_name: str) -> dict[str, Any]:
    """Fetch model version info from MLflow by model name
    and convert to dict with expected fields.

    Parameters
    ----------
    model_name: str
        Name of registered model in MLflow

    Raises
    ------
    INVALID_PARAMETER_VALUE if model not found in MLflow
    """
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_model_version_by_alias(model_name, alias="active")
    return __mlflow_version_to_dict(model_version)


def __get_manifest() -> dict[str, Any]:
    """Fetch model manifest from GCS.

    Example
    --------
    {
    "prod": {
        "version": 1,
        ...    },
    "stg": {
        "version": 2,
        ...    }
    }

    Returns
    -------
    dict
        Manifest data, or default manifest with version -1 if error occurs (to force reloading from MLflow)
    """
    try:
        # Init GCS client and download manifest.json to a temp file
        client = storage.Client()
        bucket = client.bucket("banking_model_assets")
        blob = bucket.blob("manifest.json")
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            blob.download_to_filename(tmp.name)
            tmp_path = tmp.name

        # Load temp file to dict
        with open(tmp_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        # Clean and return
        os.remove(tmp_path)
        return manifest_data

    # Log error and return default manifest with version -1 to force reloading from MLflow
    except Exception as e:
        logger.error(f"Error fetching model manifest from GCS: {e}")
        return {"prod": {"version": -1}, "stg": {"version": -1}, "dev": {"version": -1}}


def copy_model_artifacts_to_gcs(env: str) -> None:
    """Main function to sync model artifacts from MLflow to GCS if new version is detected.

    The MLflow artifactory may be offline due to cost cutting measures,
    so we want to have the latest (active) model artifacts available in GCS for the API to load from.

    The GCS bucket structure is:
    banking_model_assets/
        model/
            <env>/  # dev / stg / prod
                <version>/ # version number from MLflow, e.g. 1, 2, 3
                    # model files here, e.g. model.pkl, conda.yaml, etc.
        manifest.json  # JSON file containing current model versions in prod and stg, e.g. {"prod": {"version": 2, ...}, "stg": {"version": 3, ...}}

    Parameters
    ----------
    env: str
        Environment name (dev, stg, prod)
    """
    assert env in [
        "dev",
        "stg",
        "prod",
    ], f"Invalid environment: {env}. Must be one of dev, stg, prod."

    # 1) Fetch current model versions from MLflow
    mlflow_version_info = __get_active_model_version(f"BankingModel-{env}")
    logger.info(
        f"""Current MLflow {env} model version: v({mlflow_version_info['version']}) sha({mlflow_version_info['commit_sha']})"""
    )

    # 2) Fetch current manifest from GCS
    manifest = __get_manifest()
    logger.info(
        f"""Current manifest {env} model version: v({manifest[env]['version']})"""
    )

    # 3) Compare versions and update GCS if needed
    if mlflow_version_info["version"] != manifest[env]["version"]:
        logger.info(f"New {env} version detected. Refreshing artifacts in GCS...")
        __upsert_model_artifacts(
            mlflow_version_info["source"], str(mlflow_version_info["version"]), env=env
        )

        manifest[env] = mlflow_version_info
        __upload_manifest(manifest)
        logger.info("Model manifest updated in GCS.")

    else:
        logger.info("No changes in model versions. No update needed.")
