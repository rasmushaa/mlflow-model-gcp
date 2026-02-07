import datetime
import os
import unittest
from unittest.mock import patch

import duckdb
import pandas as pd

from src.loader import DataLoader


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Init the runtime env used in the Loader
        os.environ["GCP_PROJECT_ID"] = "MyProject"
        os.environ["GCP_LOCATION"] = "europe-west1"
        os.environ["GCP_BQ_DATASET"] = "MyDataset"

    def test_load_return_value(self):
        """Test that the load method returns the expected DataFrame with correct filtering and formatting"""
        loader = DataLoader(start_date="2000-01-01", end_date="2999-12-31")
        with patch(
            "src.data.loader.pandas_gbq.read_gbq",
            side_effect=self.__query_mock_database,
        ):
            result = loader.load()

        # Expect the None row to be dropped
        self.assertEqual(len(result), 3, "The None row should be dropped")

        # Dates should be converted to datetime.date objects
        self.assertIsInstance(result.loc[0, "date"], datetime.date)
        self.assertEqual(result.loc[0, "date"], datetime.date(2020, 1, 1))
        self.assertEqual(result.loc[1, "date"], datetime.date(2020, 1, 2))

        # Index should be reset to 0..n-1
        self.assertListEqual(list(result.index), [0, 1, 2])

    def test_load_no_limits(self):
        """Test that no date limits returns all rows"""
        loader = DataLoader()
        with patch(
            "src.data.loader.pandas_gbq.read_gbq",
            side_effect=self.__query_mock_database,
        ):
            result = loader.load()
        self.assertEqual(len(result), 3, "All rows should be queryed")

    def test_load_start_limit(self):
        """Test that inclusive start date is applied correctly"""
        loader = DataLoader(start_date="2020-01-02")
        with patch(
            "src.data.loader.pandas_gbq.read_gbq",
            side_effect=self.__query_mock_database,
        ):
            result = loader.load()
        self.assertEqual(len(result), 2, "Only the second, and last should be returned")

    def test_load_end_limit(self):
        """Test that the inclusive end date filter works"""
        loader = DataLoader(end_date="2020-01-02")
        with patch(
            "src.data.loader.pandas_gbq.read_gbq",
            side_effect=self.__query_mock_database,
        ):
            result = loader.load()
        self.assertEqual(len(result), 2, "Only the first and second should be returned")

    def __query_mock_database(
        self, query: str, project_id: str, location: str, progress_bar_type
    ):
        """Run actual querys on mocked DB

        The env variables are picked by the loader to query
        specific shemas, and those are align with the mocked table
        in the tests.
        """
        mock_df = pd.DataFrame(
            {
                "KeyDate": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"],
                "Receiver": ["A", "B", "C", "D"],
                "Amount": [100, 200, 300, 400],
                "Category": ["x", "y", "z", None],
            }
        )

        con = duckdb.connect()
        con.register("f_transactions", mock_df)
        con.execute(f"CREATE SCHEMA {os.getenv('GCP_BQ_DATASET')}")
        con.execute(
            f"CREATE TABLE {os.getenv('GCP_BQ_DATASET')}.f_transactions AS SELECT * FROM f_transactions"
        )

        return con.execute(query).df()


if __name__ == "__main__":
    unittest.main(verbosity=2)
