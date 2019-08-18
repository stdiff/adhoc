from unittest import TestCase
from pathlib import Path

import numpy as np
import pandas as pd

from adhoc.processing import Inspector, VariableType, MultiConverter

test_data = Path("data/adult.csv")

class ProcessingTest(TestCase):

    def test_an_inspection(self):
        """
        check the inspection
        """
        df = pd.read_csv(test_data)

        inspector = Inspector(df, m_cats=20)

        self.assertEqual(
            inspector.result.loc["education-num", "variable"],
            VariableType.categorical.name
        )

        ## nan must be ignored
        self.assertEqual(
            inspector.result.loc["workclass", "n_unique"],
            8
        )

        self.assertEqual(
            inspector.result.loc["sex", "variable"],
            VariableType.binary.name
        )

        df["const"] = 1
        # TODO: [datetime(year=2019,month=1,day=1) + timedelta(hours=h) for h in range(360)]
        inspector = Inspector(df, m_cats=15)

        self.assertEqual(
            inspector.result.loc["const", "variable"],
            VariableType.constant.name
        )

        self.assertEqual(
            inspector.result.loc["education-num", "variable"],
            VariableType.continuous.name
        )

        ## An "object" column must always be categorical
        self.assertTrue(
            "education" in inspector.get_cats()
        )

        self.assertEqual(inspector.get_cats(),
                         ["workclass", "education", "marital-status",
                          "occupation", "relationship", "race",
                          "sex", "native-country", "label"])

        self.assertEqual(inspector.get_cons(),
                         ["age", "fnlwgt", "education-num","capital-gain",
                          "capital-loss", "hours-per-week"])


    def test_regard_as(self):
        """
        conversion of variable type
        """

        df = pd.read_csv(test_data)
        inspector = Inspector(df, m_cats=20)

        self.assertEqual(inspector.result.loc["age", "variable"],
                         VariableType.continuous.name)

        inspector.regard_as_categorical("age")
        self.assertEqual(inspector.result.loc["age", "variable"],
                         VariableType.categorical.name)

        ## If we set m_cats, then the inspection logic will be executed.
        ## As a result the manual setting will be lost.
        inspector.m_cats = 21
        self.assertEqual(inspector.result.loc["age", "variable"],
                         VariableType.continuous.name)


    def test_distribution(self):
        """
        check DataFrames for distributions
        """

        df = pd.read_csv(test_data)
        nrow = df.shape[0]
        inspector = Inspector(df, m_cats=20)

        df_cat = inspector.distribution_cats()

        self.assertAlmostEqual(
            df_cat.loc["workclass"].loc["Private", "count"] / nrow,
            df_cat.loc["workclass"].loc["Private", "rate"]
        )

        df_con = inspector.distribution_cons()

        ## Since it is just a transpose of describe(),
        ## the number of columns is equal to 8
        self.assertEqual(
            df_con.shape,
            (len(inspector.get_cons()), 8)
        )


    def test_significance(self):
        """
        Check significance tests
        """

        df = pd.read_csv(test_data)
        df_inspection = Inspector(df, m_cats=20)

        s = df_inspection.significance_test("fnlwgt","age")

        self.assertTrue(
            isinstance(s, pd.Series)
        )

        ## field1, field2, test, statistic, p-value
        self.assertEqual(len(s), 5)

        ## Default correlation
        self.assertEqual(s["test"], "Spearman correlation")

        df_pval = df_inspection.significance_test_features("label")

        self.assertEqual(df_pval.shape[1], 5)

        df_pval.set_index("field1", inplace=True)

        self.assertEqual(
            df_pval.loc["age", "test"],
            "one-way ANOVA on ranks"
        )

        self.assertEqual(
            df_pval.loc["education-num", "test"],
            "chi-square test"
        )

class TestMultiConverter(TestCase):
    def test_multi_converter(self):
        """
        test the methods of MultiConverter
        """

        df = pd.DataFrame({
            "col1": ["a", "a", "b", "c", np.nan],  ## drop is given
            "col2": [1, 2, np.nan, 2, 5],
            "col3": ["x", np.nan, "y", np.nan, "x"], ## binary
            "col4": ["s", "s", "s", "t", "u"]    ## drop is not given
        }, columns=["col1","col2","col3","col4"])

        cats = ["col1","col3","col4"]
        strategies = {cat:"most_frequent" for cat in cats}
        strategies["col1"] = "c"

        converter = MultiConverter(
            columns=df.columns,
            strategy = strategies,
            cats = cats,
            drop = {"col1":"b"}
        )

        converter.fit(df.values)  ## fit by ndarray
        converter.fit(df) ## fit by DataFrame

        dg_columns = ["col1_a","col1_c","col2","col3_y","col4_t","col4_u"]
        dg = pd.DataFrame({
            "col1_a": [1,1,0,0,0],
            "col1_c": [0,0,0,1,1],
            "col2": [1,2,(1+2+2+5)/4,2,5],
            "col3_y": [0,0,1,0,0],
            "col4_t": [0,0,0,1,0],
            "col4_u": [0,0,0,0,1]
        }, columns=dg_columns)

        self.assertEqual(converter.classes_, dg_columns)
        self.assertTrue(np.array_equal(converter.transform(df),dg.values))
