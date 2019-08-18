"""
unittest for adhoc/utilities.py
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn import datasets

from adhoc.utilities import bunch2dataframe
from adhoc.utilities import load_iris, load_boston, load_breast_cancer, load_diabetes
from adhoc.utilities import bins_by_tree

import warnings
warnings.filterwarnings("ignore")

class TestUtilities(TestCase):

    def setUp(self):
        np.random.seed(1)


    def test_buynch2dataframe(self):
        """unittest for bunch2dataframe"""
        data = datasets.load_iris()
        target = "label"
        df = bunch2dataframe(data, target=target)

        self.assertTrue(isinstance(df,pd.DataFrame))
        self.assertEqual(list(df.columns)[-1], target)


    def test_load_iris(self):
        """unittest for load_iris"""
        target = "species"
        cats = {"setosa","versicolor","virginica"}
        df = load_iris(target=target)

        self.assertTrue(isinstance(df,pd.DataFrame))
        self.assertEqual((150,5), df.shape)
        self.assertEqual(target, list(df.columns)[-1])
        self.assertEqual(cats,set(df[target].unique()))


    def test_load_boston(self):
        """unittest for load_iris"""
        target = "target"
        df = load_boston(target=target)

        self.assertTrue(isinstance(df,pd.DataFrame))
        self.assertEqual((506,14), df.shape)
        self.assertEqual(target, list(df.columns)[-1])


    def test_load_breast_cancer(self):
        """unittest for load_breast_cancer"""
        target = "label"
        df = load_breast_cancer(target=target)

        self.assertTrue(isinstance(df,pd.DataFrame))
        self.assertEqual((569,31), df.shape)
        self.assertEqual(target, list(df.columns)[-1])


    def test_load_diabetes(self):
        """unittest for load_boston"""
        target = "label"
        df = load_diabetes(target=target)

        self.assertTrue(isinstance(df,pd.DataFrame))
        self.assertEqual((442,11), df.shape)
        self.assertEqual(target, list(df.columns)[-1])


    def test_bins_by_tree(self):
        """unittest for bins_by_tree"""
        df = load_iris(target="species")
        cols = list(df.columns)
        n_bins = 3

        bins = bins_by_tree(df, field=cols[2], target=cols[3],
                            target_is_continuous=True,
                            n_bins=n_bins, n_points=200, precision=1)
        cats = sorted(bins.unique())

        self.assertTrue(isinstance(bins,pd.Series))
        self.assertEqual(len(bins.unique()), n_bins)

        for cat in cats:
            self.assertTrue(isinstance(cat,pd.Interval))
            self.assertEqual(cat.closed,"right")

        self.assertEqual(cats[0].left, -np.inf)
        self.assertEqual(cats[-1].right, np.inf)
