import logging
from data.loader import DataLoader
from utils.setup_logging import setup_logging
from utils.mlflow import ExperimentManager
from utils.context import Context
from utils.package import build_wheel
from utils.ml.metrics import classification_report_metrics, prediction_report_metrics
from utils.ml.processing import split_data
from polymodel.factory import pipeline_factory

setup_logging(level=logging.INFO)


def main():

    context = Context()
    manager = ExperimentManager()
    loader = DataLoader(**context['query'])
    pipeline = pipeline_factory(context['model'], context['transformer'])

    with manager.start_run():

        # Log all config parameters here
        manager.log_params(context.ravel()) 

        # Load data, and log as input artifact
        data = loader.load()
        manager.log_input(data, 'Raw')

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(data, **context['training'])

        # Train model pipeline
        pipeline.fit(X_train, y_train)

        # Log component features
        manager.log_dict(pipeline.features, 'layers')
        manager.log_text(str(pipeline), 'pipeline')

        # Evaluate model probabilistic predictions
        y_pred_prob = pipeline.predict_proba(X_test)
        metrics, plots = prediction_report_metrics(y_test, y_pred_prob, pipeline.model.classes)
        manager.log_metrics(metrics)
        manager.log_figures(plots)

        # Evaluate model hard predictions
        y_pred = pipeline.predict(X_test)
        metrics, plots = classification_report_metrics(y_test, y_pred)
        manager.log_metrics(metrics)
        manager.log_figures(plots)

        # Log model with with example data
        wheel_path = build_wheel()
        manager.log_model(pipeline, data_example=X_test, wheel_path=wheel_path)


if __name__ == "__main__":
    main()
