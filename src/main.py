import logging

from polymodel.pipeline import Pipeline
from src.context import Context
from src.debug import setup_logging
from src.experiment import ExperimentManager
from src.loader import DataLoader
from src.optimise import optimize_model

setup_logging(level=logging.INFO, suppress_external=True)

logger = logging.getLogger(__name__)


def main():

    context = Context("config.yaml")
    manager = ExperimentManager()
    loader = DataLoader(**context["query"])

    with manager.start_run():

        # Log all config parameters using dot notation
        manager.log_params(context.ravel(exclude_keys=["metrics"]))

        # Load and log data
        data = loader.load()
        manager.log_input(data, "Raw-data")

        # Optimize and evaluate model
        results = optimize_model(pipeline_cls=Pipeline, data=data, context=context)

        # Log model
        manager.log_model(
            results.model,
            data_example=data.drop(columns=context["training"]["target_column"]),
        )

        # Log results
        manager.log_figures(results.metrics["figures"])
        manager.log_metrics(results.metrics["values"])
        for metric, history in results.metrics["steps"].items():
            for i, value in enumerate(history):
                manager.log_metric(metric, value, step=i)
        for i, loss in enumerate(results.loss_history):
            manager.log_metric("optimization_loss", loss, step=i)

        # Log best hyperparameters
        manager.log_params(
            {f"best_hyperparams.{k}": v for k, v in results.best_params.items()}
        )


if __name__ == "__main__":
    main()
