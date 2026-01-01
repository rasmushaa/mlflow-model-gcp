import logging
from data.loader import DataLoader
from utils.setup_logging import setup_logging
from utils.mlflow import ExperimentManager
from utils.context import Context
from utils.package import build_wheel
from utils.ml.metrics import kfold_report_metrics, evaluate_model
from utils.ml.processing import kfold_iterator
from polymodel.factory import pipeline_factory

setup_logging(level=logging.DEBUG)


def load_and_log_data(manager: ExperimentManager, loader: DataLoader):
    data = loader.load()
    manager.log_input(data, 'Raw')
    return data


def fit_new_pipeline(X_train, y_train, context):
    pipeline = pipeline_factory(context['model'], context['transformer'])
    pipeline.fit(X_train, y_train)
    return pipeline


def main():

    context = Context()
    manager = ExperimentManager()
    loader = DataLoader(**context['query'])

    with manager.start_run():

        # Log all config parameters here
        manager.log_params(context.ravel()) 

        # Load and log data
        data = load_and_log_data(manager, loader)

        # K-Fold Cross Validation
        kfold_data = {}
        for fold, X_train, X_test, y_train, y_test in kfold_iterator(data, **context['training']):
            logging.info(f"Fold {fold}: Train shape: {X_train.shape}")

            # Build and Fit a new pipeline for each fold
            pipeline = fit_new_pipeline(X_train, y_train, context)

            # Evaluate model, and collect metrics and plots for the fold
            metrics, plots = evaluate_model(pipeline, X_test, y_test)
            kfold_data[f'fold{fold}'] = {
                'metrics': metrics,
                'plots': plots
            }

            # Log fold traces for macro metrics
            for metric_name, metric_value in metrics.items():
                if 'macro' in metric_name or 'accuracy' in metric_name:
                    manager.log_metrics({f'{metric_name}.kfold': metric_value}, step=fold)

        # Final model training on full data for model registry
        X_full = data.drop(columns=[context['training']['target_column']])
        y_full = data[context['training']['target_column']]
        pipeline = fit_new_pipeline(X_full, y_full, context)
        wheel_path = build_wheel()
        manager.log_model(pipeline, data_example=X_full, wheel_path=wheel_path)

        # Log component features
        manager.log_dict(pipeline.features, 'layers')
        manager.log_text(str(pipeline), 'pipeline')
        manager.log_params({'architecture': pipeline.architecture})

        # Aggregate K-Fold results for individual folds
        # Metrics has to be logged after model logging to avoid duplication issues due this Bug: https://github.com/ecmwf/anemoi-core/issues/190
        metrics, plots = kfold_report_metrics(kfold_data) # Aggregate metrics, and concatenate plots
        manager.log_figures(plots)
        manager.log_metrics(metrics)

if __name__ == "__main__":
    main()
