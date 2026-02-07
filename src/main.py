import logging

from metrics_toolbox import EvaluatorBuilder

from context import Context
from experiment import ExperimentManager
from loader import DataLoader
from ml.processing import kfold_iterator
from polymodel.factory import pipeline_factory
from setup_logging import setup_logging

setup_logging(level=logging.INFO, suppress_external=True)


def main():

    context = Context()
    manager = ExperimentManager()
    loader = DataLoader(**context["query"])
    evaluator = EvaluatorBuilder().from_dict(context["metrics"]).build()

    with manager.start_run():

        # Log all config parameters using dot notation
        manager.log_params(context.ravel(exclude_keys=["metrics"]))

        # Load and log data
        data = loader.load()
        manager.log_input(data, "Raw")

        # K-Fold Cross Validation
        for fold, X_train, X_test, y_train, y_test in kfold_iterator(
            data, **context["training"]
        ):
            logging.info(f"Fold {fold}: Train shape: {X_train.shape}")

            # Build and Fit a new pipeline for each fold
            pipeline = pipeline_factory(context["model"], context["transformer"])
            pipeline.fit(X_train, y_train)

            # Evaluate model
            evaluator.add_model_evaluation(model=pipeline, X=X_test, y_true=y_test)

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

        # Metrics has to be logged after model logging to avoid duplication issues due this Bug: https://github.com/ecmwf/anemoi-core/issues/190
        results = evaluator.get_results()
        manager.log_figures(results["figures"])
        manager.log_metrics(results["values"])
        for metric, history in results["steps"].items():
            for i, value in enumerate(history):
                manager.log_metric(metric, value, step=i)


if __name__ == "__main__":
    main()
