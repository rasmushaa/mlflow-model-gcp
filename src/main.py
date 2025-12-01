from data.loader import DataLoader
from utils.mlflow import ExperimentManager
from utils.context import Context
from utils.ml.metrics import classification_report_metrics, prediction_report_metrics
from utils.ml.processing import split_data
from polymodel.model.factory import create_model
from polymodel.preprocessor.factory import create_preprocessors


def main():

    context = Context()
    manager = ExperimentManager()
    loader = DataLoader(**context.config['query'])


    with manager.start_run():

        # Log all config parameters here
        manager.log_params(context.config_flat) 

        # Load data, and log as input artifact
        data = loader.load()
        manager.log_input(data, 'Raw')

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = split_data(data, **context.config['training'])

        # Create and apply preprocessors dictionary
        preprocessors = create_preprocessors(context.config['preprocessor'])
        for key, preprocessor in preprocessors.items():
            manager.log_input(X_train, f'P{key}_{preprocessor.__class__.__name__}_input')
            X_train = preprocessor.fit_transform(X_train, y_train)
            X_test = preprocessor.transform(X_test)

        # Log actual model training data
        manager.log_input(X_train, f'Model_X_train')
        manager.log_input(X_test, f'Model_X_test')
        manager.log_input(y_train, f'Model_y_train')
        manager.log_input(y_test, f'Model_y_test')

        # Create model, and train
        model = create_model(**context.config['model'])
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred_prob = model.predict_proba(X_test)
        metrics, plots = prediction_report_metrics(y_test, y_pred_prob, model.classes_)
        manager.log_metrics(metrics)
        manager.log_figures(plots)

        y_pred = model.predict(X_test)
        metrics, plots = classification_report_metrics(y_test, y_pred)
        manager.log_metrics(metrics)
        manager.log_figures(plots)

        # Log model with preprocessors
        manager.log_model(model, preprocessors, input_example=X_train.iloc[:1, :])


if __name__ == "__main__":
    main()
