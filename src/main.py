import logging

from context import Context
from experiment import ExperimentManager
from loader import DataLoader
from ml.metrics import evaluate_model, kfold_report_metrics
from ml.processing import kfold_iterator
from polymodel.factory import pipeline_factory
from setup_logging import setup_logging

setup_logging(level=logging.INFO)


def main():

    context = Context()
    manager = ExperimentManager()
    loader = DataLoader(**context["query"])

    with manager.start_run():

        # Log all config parameters using dot notation
        manager.log_params(context.ravel())

        # Load and log data
        data = loader.load()
        manager.log_input(data, "Raw")

        # K-Fold Cross Validation
        kfold_data = {}
        for fold, X_train, X_test, y_train, y_test in kfold_iterator(
            data, **context["training"]
        ):
            logging.info(f"Fold {fold}: Train shape: {X_train.shape}")

            # Build and Fit a new pipeline for each fold
            pipeline = pipeline_factory(context["model"], context["transformer"])
            pipeline.fit(X_train, y_train)

            # Evaluate model, and collect metrics and plots for the fold
            metrics, plots = evaluate_model(pipeline, X_test, y_test)
            kfold_data[f"fold{fold}"] = {"metrics": metrics, "plots": plots}

            # Log fold traces to detect overfitting on the main metrics
            for metric_name, metric_value in metrics.items():
                if "macro" in metric_name or "accuracy" in metric_name:
                    manager.log_metrics(
                        {f"{metric_name}.kfold": metric_value}, step=fold
                    )

        # Final model training on full data for model registry
        X_full = data.drop(columns=[context["training"]["target_column"]])
        y_full = data[context["training"]["target_column"]]
        pipeline = pipeline_factory(context["model"], context["transformer"])
        pipeline.fit(X_full, y_full)
        manager.log_model(pipeline, data_example=X_full)

        # Log pipeline architecture and layers
        manager.log_dict(pipeline.layers, "layers")
        manager.set_tags({"model.architecture": pipeline.architecture})
        manager.set_tags({"model.signature": pipeline.signature})
        manager.set_tags({"model.features": pipeline.features})

        # Aggregate K-Fold results for individual folds
        # Metrics has to be logged after model logging to avoid duplication issues due this Bug: https://github.com/ecmwf/anemoi-core/issues/190
        metrics, plots = kfold_report_metrics(kfold_data)
        manager.log_figures(plots)
        manager.log_metrics(metrics)


if __name__ == "__main__":
    main()
