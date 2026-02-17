import pandas as pd

from polymodel.pipeline import Pipeline


def test_pipeline():

    # Init actual pipeline from config
    pipeline_config = {
        "text_cleaner": {
            "features": ["text"],
            "transform_mode": "inplace",
            "transform_suffix": "_cleaned",
            "hyperparams": {},
        },
        "text_vectorizer": {
            "features": ["*_cleaned"],
            "transform_mode": "inplace",
            "transform_suffix": "_vec",
            "hyperparams": {"kbest": 2},
        },
        "random_forest": {
            "features": ["*_vec"],
            "transform_mode": "replace",
            "transform_suffix": "DoesNotMatterAtEnd",
            "hyperparams": {"n_estimators": 100, "max_depth": 5},
        },
    }
    pipeline = Pipeline.from_config(pipeline_config)

    # Fit the pipeline on some dummy data
    df_train = pd.DataFrame(
        {
            "text": ["Hello world", "Machine learning is great", "I love programming"],
            "extra": [1, 2, 3],
            "target": [0, 1, 0],
        }
    )
    pipeline.fit(df_train.drop(columns=["target"]), df_train["target"])

    # Make predictions
    df_test = pd.DataFrame(
        {"text": ["Hello", "I enjoy machine learning"], "extra": [4, 5]}
    )
    _ = pipeline.predict(df_test)
    _ = pipeline.predict_proba(df_test)

    print(pipeline)
    print(pipeline.layers)
    print(pipeline.resolved_features)
    print(pipeline.classes)

    # Assertions to verify the pipeline's behavior
    assert pipeline.resolved_features == [
        "text"
    ], "Only the text input signature is used by the pipeline"
    assert pipeline.classes == [0, 1], "The model should have two classes: 0 and 1"

    # Check that the pipeline layers are correctly resolved
    assert len(pipeline.layers) == 3, "Pipeline should have three layers"
    assert pipeline.layers[0]["signature"] == [
        "text",
        "extra",
    ], "First layer should detect all input features"
    assert pipeline.layers[1]["signature"] == ["extra", "text_cleaned"]
    assert (
        len(pipeline.layers[2]["signature"]) == 3
    ), "Third layer should use all kbest features, and the original extra feature"
    assert all(
        feature.endswith("_vec") for feature in pipeline.layers[2]["resolved_features"]
    ), "Third layer should use features ending with '_vec'"
