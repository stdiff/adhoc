"""
This script is used to retrieve a data set for unit tests.
"""
from unittest import TestCase, main
from pathlib import Path
import hashlib

import pandas as pd

data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
names = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
         "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
         "hours-per-week", "native-country", "label"]
csv_path = "../data/adult.csv"
checksum = "ee2d7503652f28a713aa6a054f5d9bb610a160afb8b817e6347d155c80af9795"


class TestData(TestCase):
    def test_data(self):
        data_path = Path(__file__).parent.joinpath(csv_path)

        if not data_path.exists():
            ## if there is no data file, then we have to download the data
            df = pd.read_csv(data_url, na_values="?", names=names, skipinitialspace=True)
            df.to_csv(data_path, index=False)

        with data_path.open("rb") as fo:
            checksum_calculated = hashlib.sha256(fo.read()).hexdigest()

        self.assertEqual(checksum_calculated, checksum)


if __name__ == "__main__":
    main()