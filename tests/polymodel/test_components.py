import pandas as pd
import pytest

from polymodel.components.models.naive_bayes import NaiveBayesModel
from polymodel.components.models.random_forest import RandomForestModel
from polymodel.components.models.xgboost import XGBoostModel
from polymodel.components.transformers.date_signal import DateSignal
from polymodel.components.transformers.text_cleaner import TextCleaner
from polymodel.components.transformers.text_vectorizer import TextVevtorizer


def test_text_cleaner():
    data = {
        "text": [
            "Hello, World! 123",
            "This is a test.",
            "   Leading and trailing spaces   ",
            "Multiple   spaces",
            "Single-letter tokens: a b c d e",
            None,
        ]
    }
    df = pd.DataFrame(data)

    cleaner = TextCleaner(
        features=["text"], transform_mode="replace", transform_suffix="_new"
    )
    cleaner.fit(df)
    transformed_df = cleaner.transform(df)
    print(transformed_df)

    expected_data = {
        "text_new": [
            "hello world",
            "this is test",
            "leading and trailing spaces",
            "multiple spaces",
            "singleletter tokens",
            "",
        ]
    }
    expected_df = pd.DataFrame(expected_data)
    print(expected_df)

    assert transformed_df.equals(expected_df)
    assert cleaner.signature == ["text"]
    assert cleaner.resolved_features == ["text"]


def test_text_count_vector():
    data = {
        "text": [
            "hello world",
            "hello hello",
            "world",
            "test",
            "hello test",
            None,
        ],
        "target": [1, 1, 1, 1, 1, 0],
    }
    df = pd.DataFrame(data)

    vectorizer = TextVevtorizer(
        features=["*"], transform_mode="replace", transform_suffix="_vec", kbest=2
    )
    vectorizer.fit(df.drop(columns=["target"]), df["target"])
    transformed_df = vectorizer.transform(df.drop(columns=["target"]))
    print(transformed_df)

    expected_data = {
        "hello_vec": [1.0, 2.0, 0.0, 0.0, 1.0, 0.0],
        "world_vec": [1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    }
    expected_df = pd.DataFrame(expected_data)
    print(expected_df)

    assert transformed_df.equals(expected_df)
    assert vectorizer.signature == ["text"]
    assert vectorizer.resolved_features == ["text"]


def test_random_forest_model():
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "extra": [10, 20, 30, 40, 50],
        "target": [0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    model = RandomForestModel(
        features=["feature*"],
        transform_mode="replace",
        transform_suffix="_pred",
        n_estimators=10,
    )
    model.fit(df.drop(columns=["target"]), df["target"])
    predictions = model.predict(df.drop(columns=["target"]))
    print(predictions)

    assert len(predictions) == len(df)
    assert model.classes == [0, 1]
    assert model.signature == ["feature1", "feature2", "extra"]
    assert model.resolved_features == ["feature1", "feature2"]  # Resolved features


def test_naive_bayes_model():
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "extra": [10, 20, 30, 40, 50],
        "target": [0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    model = NaiveBayesModel(
        features=["feature*"],
        transform_mode="replace",
        transform_suffix="_pred",
    )
    model.fit(df.drop(columns=["target"]), df["target"])
    predictions = model.predict(df.drop(columns=["target"]))
    print(predictions)

    assert len(predictions) == len(df)
    assert model.classes == [0, 1]
    assert model.signature == ["feature1", "feature2", "extra"]
    assert model.resolved_features == ["feature1", "feature2"]


def test_date_signal():
    data = {
        "date": [
            "2020-09-15",
            None,
        ]
    }
    df = pd.DataFrame(data)

    transformer = DateSignal(
        features=["date"], transform_mode="replace", transform_suffix="_sig"
    )
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    print(transformed_df)

    pytest.approx(
        transformed_df["date_year_sin_sig"].iloc[0], 0.01
    ) == -1  # September is month 9, so sin(2*pi*9/12) = -1
    pytest.approx(
        transformed_df["date_year_cos_sig"].iloc[0], 0.01
    ) == 0  # cos(2*pi*9/12) = 0
    pytest.approx(
        transformed_df["date_month_sin_sig"].iloc[0], 0.01
    ) == 0.0  # Day 15, so sin(2*pi*15/31) should be close to 0
    pytest.approx(
        transformed_df["date_month_cos_sig"].iloc[0], 0.01
    ) == -1  # cos(2*pi*15/31) should be close to -1
    pytest.approx(
        transformed_df["date_dayofweek_sig"].iloc[0], 0.01
    ) == 1  # September 15, 2020 was a Tuesday (dayofweek=1)
    pytest.approx(
        transformed_df["date_is_weekend_sig"].iloc[0], 0.01
    ) == 0  # Tuesday is not a weekend
    pytest.approx(
        transformed_df["date_year_sin_sig"].iloc[1], 0.01
    ) == -1  # Missing date treated as 1970-01-01, which is month 1


def test_xgboost_model():
    data = {
        "feature1": [1, 2, 3, 4, 5],
        "feature2": [5, 4, 3, 2, 1],
        "extra": [10, 20, 30, 40, 50],
        "target": [0, 0, 1, 1, 1],
    }
    df = pd.DataFrame(data)

    model = XGBoostModel(
        features=["feature*"],
        transform_mode="replace",
        transform_suffix="_pred",
        n_estimators=10,
    )
    model.fit(df.drop(columns=["target"]), df["target"])
    predictions = model.predict(df.drop(columns=["target"]))
    probs = model.predict_proba(df.drop(columns=["target"]))
    print(predictions)
    print(probs)

    assert len(predictions) == len(df)
    assert probs.shape == (len(df), 2)  # Should have probabilities for both classes
    assert model.classes == [0, 1]
    assert model.signature == ["feature1", "feature2", "extra"]
    assert model.resolved_features == ["feature1", "feature2"]
