"""
unittest for adhoc/modeling.py
"""

from unittest import TestCase

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from adhoc.modeling import add_prefix_to_param, simple_pipeline_cv
from adhoc.modeling import cv_results_summary, pick_the_last_estimator
from adhoc.modeling import show_coefficients, show_feature_importance
from adhoc.modeling import recover_label, ROCCurve
from adhoc.utilities import load_boston, load_iris, load_breast_cancer

class ModelingTest(TestCase):
    iris_X = None
    iris_y = None
    iris_plr = None
    boston_X = None
    boston_y = None
    boston_tree = None
    boston_enet = None
    breast_cancer_X = None
    breast_cancer_y = None
    breast_cancer_plt = None

    @classmethod
    def setUpClass(cls) -> None:
        np.random.seed(1)

        ## iris (multi-class classification)
        df = load_iris(target="label")
        df = df.sample(frac=1,replace=False)
        cls.iris_X = df.drop("label", axis=1)
        cls.iris_y = df["label"]

        cls.iris_plr = GridSearchCV(
            LogisticRegression(solver="liblinear", multi_class="auto"),
            param_grid={"C":[0.1,1]},
            cv=3, return_train_score=True)
        cls.iris_plr.fit(cls.iris_X,cls.iris_y)

        ## boston dataset (regression)
        ## TODO: use different data set
        target = "price"
        df = load_boston(target=target)
        df = df.sample(frac=1,replace=False)
        cls.boston_X = df.drop("price", axis=1)
        cls.boston_y = df["price"]

        cls.boston_tree = GridSearchCV(
            DecisionTreeRegressor(),
            param_grid={"max_depth":[3,5,7]},
            cv=5, scoring="neg_mean_squared_error",
            return_train_score=False)
        cls.boston_tree.fit(cls.boston_X, cls.boston_y)

        cls.boston_enet = GridSearchCV(
            ElasticNet(),
            param_grid={"alpha": [0.1,1.0], "l1_ratio": [0.2,0.5,0.8]},
            cv=3, scoring="neg_mean_squared_error",
            return_train_score=True)
        cls.boston_enet.fit(cls.boston_X, cls.boston_y)

        ## breast (binary classification)
        df = load_breast_cancer("label")
        cls.breast_cancer_X = df.drop("label", axis=1)
        cls.breast_cancer_y = df["label"]

        cls.breast_cancer_plr = GridSearchCV(
            LogisticRegression(solver="liblinear"),
            param_grid={"C":[0.1,1]}, cv=3)
        cls.breast_cancer_plr.fit(cls.breast_cancer_X, cls.breast_cancer_y)


    def setUp(self) -> None:
        np.random.seed(1)

    def test_add_prefix_to_param(self):
        """unittest for add_prefix_to_param"""
        pref = "YourPrefix"
        before = {"param_1": [1,2,3],
                  "param_2": [5,9,1]}
        correct_keys = ["YourPrefix__param_1","YourPrefix__param_2"]

        after = add_prefix_to_param(pref, before)

        self.assertTrue(isinstance(after,dict))
        self.assertEqual(correct_keys, sorted(after.keys()))
        self.assertEqual(after[correct_keys[0]], before["param_1"])
        self.assertEqual(after[correct_keys[1]], before["param_2"])


    def test_simple_pipeline_cv(self):
        """unittest for simple_pipeline_cv"""
        plr = LogisticRegression(solver="liblinear")
        name = "prefix"
        cv = 11
        param_grid = {"C": [2,3,5]}
        model = simple_pipeline_cv(name=name,
                                   model=plr,
                                   param_grid=param_grid,
                                   scaler=MinMaxScaler(),
                                   cv=cv)
        pipeline = model.estimator

        self.assertTrue(isinstance(model,GridSearchCV))
        self.assertTrue(isinstance(pipeline,Pipeline))
        self.assertEqual("scaler", pipeline.steps[0][0])
        self.assertTrue(isinstance(pipeline.steps[0][1],
                                   MinMaxScaler))
        self.assertEqual(name, pipeline.steps[1][0])
        self.assertTrue(isinstance(pipeline.steps[1][1],
                                   LogisticRegression))
        self.assertEqual(cv, model.cv)
        for key in model.param_grid.keys():
            self.assertTrue(key.startswith(name+"__"))


    def test_cv_results_summary(self):
        """unittest for cv_results_summary"""
        results = cv_results_summary(self.boston_tree, alpha=0.05)

        self.assertTrue(isinstance(results,pd.DataFrame))

        cols = ["mean_test_score", "std_test_score",
                "test_CI_low", "test_CI_high"]

        self.assertEqual((3,len(cols)+1), results.shape)
        self.assertEqual(cols, list(results.columns)[:len(cols)])
        self.assertEqual("rank_test_score", results.index.name)

        self.assertEqual(1, (results["test_CI_low"] < results["mean_test_score"]).mean())
        self.assertEqual(1, (results["mean_test_score"] < results["test_CI_high"]).mean())

        delta_left = results["mean_test_score"] - results["test_CI_low"]
        delta_right = results["test_CI_high"] - results["mean_test_score"]
        self.assertAlmostEqual(0, (delta_left - delta_right).mean())


    def test_cv_results_summary2(self):
        ## including
        results = cv_results_summary(self.iris_plr, alpha=0.05)
        self.assertTrue("mean_train_score" in results.columns)


    def test_pick_the_last_estimator(self):
        """unittest for pick_the_last_estimator"""
        ## Case 1) Pipeline
        model1 = simple_pipeline_cv(
            name="plr",
            model=LogisticRegression(solver="liblinear",
                                     multi_class="auto"),
            param_grid={"C":[0.1,1]}, cv=3)
        model1.fit(self.iris_X,self.iris_y)
        estimator1 = pick_the_last_estimator(model1)

        self.assertTrue(isinstance(estimator1,LogisticRegression))

        ## Case 2) Single model
        estimator2 = pick_the_last_estimator(self.iris_plr)
        self.assertTrue(isinstance(estimator2,LogisticRegression))


    def test_show_coefficients_classification(self):
        """unittest for show_coefficients"""
        reg_coef = show_coefficients(self.iris_plr,
                                     self.iris_X.columns)

        self.assertIsInstance(reg_coef, pd.DataFrame)
        self.assertEqual((5,3), reg_coef.shape)
        self.assertTrue("intercept", reg_coef.index[4])

        ## raise an Exception if coef_ attribute is not available.
        with self.assertRaises(Exception):
            show_coefficients(self.boston_tree, self.boston_X.columns)


    def test_show_coefficients_regression(self):
        reg_coef = show_coefficients(self.boston_enet,
                                     self.boston_X.columns)

        self.assertIsInstance(reg_coef, pd.DataFrame)
        self.assertEqual((self.boston_X.shape[1]+1,1), reg_coef.shape)
        self.assertEqual(["coefficient"], reg_coef.columns.tolist())
        self.assertTrue("intercept", reg_coef.index[-1])

    def test_show_coefficients_binary_classification(self):
        reg_coef = show_coefficients(self.breast_cancer_plr,
                                     self.breast_cancer_X.columns)

        self.assertIsInstance(reg_coef, pd.DataFrame)
        self.assertEqual((self.breast_cancer_X.shape[1]+1,1), reg_coef.shape)
        self.assertEqual(["malignant"], reg_coef.columns.tolist())
        self.assertTrue("intercept", reg_coef.index[-1])


    # def test_show_tree(self):
    #     ## GridSearchCV instance
    #     image = show_tree(self.boston_tree, self.boston_X.columns)
    #     self.assertIsInstance(image, Image)
    #
    #     ## DecisionTreeRegressor
    #     tree = DecisionTreeRegressor(max_depth=3, random_state=2)
    #     tree.fit(self.boston_X, self.boston_y)
    #     image = show_tree(tree, self.boston_X.columns)
    #     self.assertIsInstance(image, Image)


    def test_show_feature_importance(self):
        """unittest for show_feature_importance"""
        self.assertIsInstance(self.boston_tree, GridSearchCV)

        importance = show_feature_importance(
            estimator=self.boston_tree,
            columns=self.boston_X.columns)

        self.assertIsInstance(importance, pd.Series)
        self.assertEqual(self.boston_X.shape[1], len(importance))

        ## raise an Exception if feature_importances_ attr is not available
        with self.assertRaises(AttributeError):
            show_feature_importance(estimator=self.iris_plr,
                                    columns=self.iris_X.columns)

    def test_show_feature_importance_model(self):
        model = pick_the_last_estimator(self.boston_tree)
        self.assertIsInstance(model, DecisionTreeRegressor)
        self.assertNotIsInstance(model, GridSearchCV)

        importance = show_feature_importance(
            estimator=model,
            columns=self.boston_X.columns)

        self.assertIsInstance(importance, pd.Series)
        self.assertEqual(self.boston_X.shape[1], len(importance))


    def test_recover_label(self):
        """unittest for recover_label"""
        s = self.iris_y.copy()
        s = s.apply(lambda y: "OTHER" if y == "versicolor" else y)
        Y = pd.get_dummies(self.iris_y)

        ## Case 1) without sep
        dg = recover_label(
            data=Y,
            field2columns={"SPECIES":["setosa","virginica"]},
            sep=None, other="OTHER", inplace=False)

        self.assertTrue(isinstance(dg,pd.DataFrame))
        self.assertEqual(Y.shape[1]+1, dg.shape[1])
        self.assertEqual("SPECIES", list(dg.columns)[-1])
        self.assertEqual(1, (dg["SPECIES"] == s).mean())

        ## Case 2) with sep
        Y.columns = ["FIELD|%s" % col for col in Y.columns]
        dg = recover_label(
            data=Y,
            field2columns={"FIELD":["FIELD|setosa","FIELD|virginica"]},
            sep="|", other="OTHER", inplace=False)

        self.assertTrue(isinstance(dg,pd.DataFrame))
        self.assertEqual(Y.shape[1]+1, dg.shape[1])
        self.assertEqual("FIELD", list(dg.columns)[-1])
        self.assertEqual(1, (dg["FIELD"] == s).mean())


class TestROCCurve(TestCase):
    y_true = None
    y_pred = None
    y_score = None
    roc = None

    @classmethod
    def setUpClass(cls) -> None:
        df = load_breast_cancer(target="LABEL")
        X,y = df.drop("LABEL",axis=1),df["LABEL"]

        plr = LogisticRegression(C=1.0, solver="liblinear")
        plr.fit(X,y)
        Y = pd.DataFrame(plr.predict_proba(X),
                         columns=plr.classes_)

        cls.y_true = y
        cls.y_pred = plr.predict(X)
        cls.y_score = Y["malignant"]


    def setUp(self) -> None:
        np.random.seed(1)
        self.roc = ROCCurve(y_true=self.y_true,
                            y_score=self.y_score,
                            pos_label="malignant")


    def test_ROCCurve(self):
        """unittest for ROCCurve"""
        ## auc attribute
        self.assertTrue(isinstance(self.roc.auc,float))
        self.assertTrue(self.roc.auc >= 0.5)
        self.assertTrue(self.roc.auc <= 1.0)

        ## scores attribute
        cols = ["recall", "precision", "fp_rate",
                "accuracy", "f1_score", "n_pos_pred"]
        self.assertTrue(isinstance(self.roc.scores,pd.DataFrame))
        self.assertEqual(cols, list(self.roc.scores.columns))
        self.assertEqual("threshold", self.roc.scores.index.name)


    def test_predict_thru_threshold(self):
        y_pred = self.roc.predict_thru_threshold(threshold=0.5)
        self.assertTrue(isinstance(y_pred,np.ndarray))

        y_pred = pd.Series(y_pred, index=self.y_true.index)\
                   .map({0:"benign", 1:"malignant"})

        self.assertEqual(1, (y_pred == self.y_pred).mean())


    def test_get_scores_thru_threshold(self):
        s = self.roc.get_scores_thru_threshold(0.5)
        self.assertTrue(isinstance(s,pd.Series))

        index = ["threshold", "recall", "precision",
                 "accuracy", "f1_score", "n_pos_pred",
                 "fp_rate"]

        self.assertEqual(index, list(s.index))
        self.assertEqual((self.y_true == self.y_pred).mean(),
                         s["accuracy"])
        self.assertEqual(0.5, s["threshold"])


    def test_get_confusion_matrix(self):
        conf = self.roc.get_confusion_matrix(threshold=0.5)
        self.assertTrue(isinstance(conf,pd.DataFrame))


    def test_roc_curve(self):
        # check if it is error-free
        self.roc.show_roc_curve()


    def test_show_metrics(self):
        # check if it is error-free
        self.roc.show_metrics()


    def test_optimize_expected_value(self):
        df_scoring = self.roc.optimize_expected_value(
            proportion_positive=self.roc.y_true.mean(),
            scaling=True)

        ## The original index must lie in the DataFrame
        self.assertTrue("index", df_scoring.columns)

        ## scores must be sorted in the descending order
        s = df_scoring["score"].copy()
        s.sort_values(ascending=False, inplace=True)
        self.assertEqual(1,(df_scoring["score"].values == s.values).mean())

        ## The first value of the index must be 1, not 0.
        self.assertTrue(1, df_scoring.index[0])

        ## If we do scaling, the expected values agrees at the end
        last_row = df_scoring.iloc[df_scoring.shape[0]-1,:]
        self.assertAlmostEqual(last_row["expected (score)"],
                               last_row["expected (random)"])

        self.assertEqual(self.roc.y_true.sum(), last_row["n_true"])

        ## check if it is error-free
        self.roc.show_expected_value(
            proportion_positive=self.roc.y_true.mean(),
            scaling=True
        )