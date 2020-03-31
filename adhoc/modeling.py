"""
module for mathematical modeling
"""
from typing import *

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

## for show_tree
from io import StringIO
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from IPython.display import Image
import pydot

## for ROCCurve class
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import recall_score, precision_score, f1_score
import matplotlib.pyplot as plt


## Hyperparameters for GridSearchCV
## Keep them simple. They are only for first simple analysis.
grid_params = {
    "ElasticNet": {
        "alpha": [0.01, 0.1, 1.0],
        "l1_ratio": [0.1, 0.4, 0.7, 1]
    },
    "LogisticRegression": {
        "C": [0.1, 1.0, 10],
        "l1_ratio": [0.1, 0.5, 0.9]
    },
    "DecisionTree": {
        'max_leaf_nodes': [3, 6, 12, 24]
    },
    "RandomForest": {
        'n_estimators': [10, 30, 50],
        'max_depth': [3, 5]
    },
    "XGB": {
        'learning_rate': [0.01, 0.1],
        'n_estimators': [5, 10, 20],
        'max_depth': [5, 10]}
}



def add_prefix_to_param(prefix:str, param_grid:dict) -> Dict[str,list]:
    """
    Create a param_grid for Pipeline from an "ordinary" param_grid.

    :param prefix: name of the step
    :param param_grid: ordinary grid_param
    :return: modified dict
    """
    return { "%s__%s" % (prefix,k): v for k,v in param_grid.items()}


def simple_pipeline_cv(name:str, model:BaseEstimator, param_grid:Dict[str,list],
                       cv:int=5, scoring:Any="accuracy", scaler:BaseEstimator=None,
                       return_train_score=True, **kwarg) -> GridSearchCV:
    """
    Create a pipeline with only one scaler and an estimator

    :param name: name of your model (e.g. rf).
    :param model: Estimator (Classifier/Regressor) instance
    :param param_grid: grid parameters for cross validation. Same as
    :param cv: number of folds in CV
    :param scoring: See https://scikit-learn.org/stable/modules/model_evaluation.html
    :param scaler: Transfoer instance. The default value is MinMaxScaler()
    :param return_train_score: if self.cv_results_ contains the average training scores
    :param kwarg: arguments for GridSearchCV
    :return: GridSearchCV instance
    """
    import sklearn

    if scaler is None:
        scaler = MinMaxScaler()

    pipeline = Pipeline([("scaler", scaler),  (name, model)])
    param_grid = add_prefix_to_param(name, param_grid)

    ## TODO: We remove this if-statement in future.
    if sklearn.__version__ > "0.24":
        ## new version does not have the parameter iid
        model = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, refit=True,
                             return_train_score=return_train_score, **kwarg)
    elif kwarg.get("iid") is not None:
        ## if it is explicitly given
        model = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, refit=True,
                             return_train_score=return_train_score, **kwarg)
    else:
        ## if nothing is given
        model = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, refit=True,
                             return_train_score=return_train_score, iid=False, **kwarg)
    return model


def cv_results_summary(grid:GridSearchCV, alpha:float=0.05) -> pd.DataFrame:
    """
    Make the result of CV more smaller and add confidence interval of
    cross-validation scores.

    :param grid: fitted instance of GridSearchCV
    :param alpha: significance level (default = 0.05)
    :return: DataFrame of the results of the cross-validation.
    """

    ## compute the confidence interval of validation scores
    ## Assumption: scores follow a normal distribution and its mean and standard deviation
    ## are not known.

    df = pd.DataFrame(grid.cv_results_)

    n_fold = grid.cv ## number of folds in CV
    delta = df["std_test_score"]*stats.t.ppf(1-alpha/2, n_fold-1)/np.sqrt(n_fold)
    df["test_CI_low"] = df["mean_test_score"] - delta
    df["test_CI_high"] = df["mean_test_score"] + delta

    ## select columns
    param_list = [k for k in grid.cv_results_.keys() if k.startswith("param_")]

    cols = ["rank_test_score", "mean_test_score", "std_test_score",
        "test_CI_low", "test_CI_high"]

    if grid.return_train_score:
        cols.append("mean_train_score")

    cols.extend(param_list)

    df = df[cols].copy()
    df.set_index(cols[0], inplace=True)

    return df.sort_index()


def pick_the_last_estimator(grid:GridSearchCV) -> BaseEstimator:
    """
    Pick the "last" component of the Pipeline instance in the given GridSearchCV instance.
    If the estimator of the GridSearchCV is not a Pipeline, then we give the estimator.

    :param grid: fitted GridSearchCV instance
    :return: estimator instance
    """
    if isinstance(grid.best_estimator_, Pipeline):
        ## pick the last estimator
        return grid.best_estimator_.steps[-1][1]
    else:
        return grid.best_estimator_


def show_coefficients(grid:GridSearchCV, columns:List[str]) -> pd.DataFrame:
    """
    show the regression coefficients of the trained model.

    WARNING: If you have a fitted Pipeline instance, then we pick the last estimator in it.
    And the coefficients you see are coefficients after other Transformers such as MinMaxScaler.
    While you can safely compare fields by looking at the coefficients, you can not interpret
    the coefficients in an original scale.

    This function works if the last component of the pipeline has .coef_ and intercept_ as
    attributes.

    :param grid: fitted GridSearchCV instance
    :param columns: list of columns
    :return: DataFrame of coefficients
    """
    ## TODO: change the name of the method, because show_* must return a figure
    model = pick_the_last_estimator(grid)

    if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
        raise Exception("You probably have no linear model")

    if hasattr(model,"classes_"):
        ## classification
        ## If we have a binary classifier, model.coef_ is an array of shape (1,p),
        ## but model.class_ contains two classes. Thus we need to pick the positive class
        labels = model.classes_ if len(model.classes_) > 2 else [model.classes_[1]]
    else:
        ## regression
        labels = ["coefficient"]

    df_coef = pd.DataFrame(model.coef_.T, index=columns, columns=labels)
    df_intercept = pd.DataFrame([model.intercept_], index=["intercept"], columns=labels)
    df_coef = pd.concat([df_coef,df_intercept], axis=0)

    return df_coef


def show_tree(model:Union[DecisionTreeRegressor, DecisionTreeClassifier, GridSearchCV],
              columns:Union[List[str],pd.Index]) -> Image:
    """
    Visualize the given trained DecisionTree model on Jupyter.
    This function requires also pydot.

    :param model: DecisionTree model or GridSearchCV instance with a Tree model
    :param columns: names of columns
    :return: Image instance (for Jupyter)
    """
    if isinstance(model, GridSearchCV):
        model = pick_the_last_estimator(model)

    dot_data = StringIO()
    export_graphviz(model, out_file=dot_data, feature_names=columns,
                    filled=True, rounded=True, special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
    return Image(graph.create_png())


def show_feature_importance(estimator:BaseEstimator,
                            columns:Union[List[str],pd.Index]) -> pd.Series:
    """
    Return the series of feature importance of given random forest model.
    XGB model and DecisionTree model can be accepted as well.

    :param estimator: fitted Estimator with `feature_importances_` attribute
    :param columns: list (or pandas.Index) of column names
    :return: Series of feature importance
    """
    if isinstance(estimator, GridSearchCV):
        model = pick_the_last_estimator(estimator)
    else:
        model = estimator

    if hasattr(model,"feature_importances_"):
        s = pd.Series(model.feature_importances_, index=columns, name="importance")
        return s.sort_values(ascending=False)
    else:
        raise AttributeError("Your model does not have an attribute 'feature_importances_'.")


def recover_label(data:pd.DataFrame, field2columns:Dict[str,List[str]],
                  sep:str=None, other:str="other", inplace=True) -> pd.DataFrame:
    """
    Similar to LabelBinarizer.inverse_transform but without the fitted instance.
    Assume you have binary columns col1 and col2. Then this function adds a column
    "field" whose values are as follows.
    - field == col1 if col1 == 1,
    - field == col2 if col2 == 1, and
    - Otherwise field == "other"

    If you give sep, then we remove field + sep from column names and use it as values.
    Namely col1 = field + sep + val1 and field == val1 if col1 == 1.

    WARNING: We do not check if col1 and col2 does not have 1 at the same time.

    :param data: DataFrame
    :param field2columns: mapping from a field name to the list of columns
    :param sep: separator character between field and value. If it is not given, then
                we use a column name as a vale.
    :param other: the value of the other values.
    :param inplace: if we modify the given DataFrame. If it is False, then we return
                    a modified DataFrame
    :return: None if inplace is True, else the modified DataFrame.
    """

    if not inplace:
        data = data.copy()

    ## check if given column names always have field + sep as prefix
    if sep is not None:
        ## sep can be an empty string: "if sep" should be avoided
        for field, columns in field2columns.items():
            for column in columns:
                if not column.startswith(field+sep):
                    raise ValueError("Column %s does not have %s%s as prefix" % (column,field,sep))
                if not column in data.columns:
                    raise ValueError("Column %s can not be found in the given data." % column)

    for field, columns in field2columns.items():
        if sep:
            values = [column[len(field + sep):] for column in columns]
        else:
            values = columns

        data[field] = data[columns[0]].map({0: other, 1: values[0]})
        for column, value in zip(columns[1:], values[1:]):
            data.loc[data[column] == 1,field] = value

    if not inplace:
        return data


class ROCCurve:
    def __init__(self, y_true:pd.Series,
                 y_score:Union[np.ndarray,pd.Series],
                 pos_label:Any=1):
        """
        This class provides API to calculate performance metrics
        which are relevant to a binary classification.

        Note that the index of y_true is used as the instance id.

        :param y_true: a Pandas Series
        :param y_score: an array-like object containing scores for
                        the positive label.
        :param pos_label: positive value in y_true.
        """

        self.y_true = np.array([1 if y == pos_label else 0 for y in y_true])
        self.y_index = y_true.index

        if isinstance(y_score, pd.Series):
            ## remove index
            self.y_score = y_score.values
        else:
            self.y_score = y_score

        self.pos_label = pos_label
        self.fpr, self.tpr, self.thresholds = roc_curve(self.y_true, self.y_score, pos_label=1)
        self.thresholds = self.thresholds[1:]
        self._scores = None


    @property
    def auc(self) -> float:
        """The area under the ROC curve"""
        return roc_auc_score(self.y_true, self.y_score)


    @property
    def scores(self) -> pd.DataFrame:
        """DataFrame of recall, precision, FP-rate, accuracy and F1 score for each threshold"""
        if self._scores is None:
            cols = ["threshold", "recall", "precision", "fp_rate",
                    "accuracy", "f1_score", "n_pos_pred"]
            rows = []

            ## TODO: find an efficient algorithm to calculate metrics at the same time
            for t in self.thresholds:
                row = self.get_scores_thru_threshold(t)
                rows.append([row[c] for c in cols])

            self._scores = pd.DataFrame(rows, columns=cols).set_index(cols[0])

        return self._scores


    def predict_thru_threshold(self, threshold:float) -> np.ndarray:
        """
        make a prediction by splitting scores by the given threshold

        :param threshold:
        :return: a binary array of predictions
        """
        return (self.y_score >= threshold).astype(int)


    def get_scores_thru_threshold(self, threshold:float=0.5) -> pd.Series:
        """
        compute the performance scores.

        n_pos_pred is the number of instances which are predicted as positive.

        :param threshold: threshold
        :return: Series of scores
        """
        y_pred = self.predict_thru_threshold(threshold)

        s = pd.Series(name="performance", dtype=np.float)
        s["threshold"] = threshold
        s["recall"] = recall_score(self.y_true, y_pred)
        s["precision"] = precision_score(self.y_true, y_pred)
        s["accuracy"] = np.mean(self.y_true == y_pred)
        s["f1_score"] =  f1_score(self.y_true, y_pred)
        s["n_pos_pred"] = y_pred.sum()

        fp = np.logical_and(self.y_true == 0, y_pred == 1).sum()
        tn_fp = np.sum(self.y_true == 0)
        ## tn_pf is never zero because it is the number of
        ## negative labels in the data but just in case
        s["fp_rate"] = fp / tn_fp if tn_fp > 0 else np.nan

        return s


    def get_confusion_matrix(self, threshold:float=0.5) -> pd.DataFrame:
        """
        returns the crosstab of the prediction and the true value
        :param threshold:
        :return: the
        """
        y_pred = self.predict_thru_threshold(threshold)

        return pd.crosstab(
            pd.Series(y_pred, name="prediction"),
            pd.Series(self.y_true, name="true value")
        )


    def show_roc_curve(self):
        """
        plot ROC curve. The AUC is shown in the title
        """
        plt.plot(self.fpr, self.tpr)
        plt.plot([0, 1], [0, 1], "k:")
        plt.title("ROC curve (AUC = %0.4f)" % self.auc)
        plt.xlabel("False-Positive Rate")
        plt.ylabel("True-Positive Rate")


    def show_metrics(self):
        """
        plot curves of metrics (recall, precision, accuracy and f1 score.
        The false-positive rate is not shown because it is the only metric
        in the DataFrame which should be smaller and therefore it is
        very confusing.
        """
        cols = ["recall", "precision", "accuracy", "f1_score"]
        self.scores[cols].plot()
        plt.title("Scores for each threshold")


    def optimize_expected_value(self,
                                proportion_positive:float=None,
                                scaling:bool=True) -> pd.DataFrame:
        """
        Compute a table for lead prioritization. That is,
        - sort the instances in the ascending order of scores (probabilities)
        - scaling the scores so that the expected value agrees with the given proportion

        :param proportion_positive: (estimated) proportion of the positive instances
                                    in the data set.
        :param scaling: if we do scaling the probabilities
        :return: DataFrame of sorted scores and expected values.
        """

        if scaling and proportion_positive is None:
            raise ValueError("proportion_positive is needed for scaling")

        df = pd.DataFrame(index=self.y_index)
        df["y_true"] = self.y_true
        df["score"] = self.y_score
        df.sort_values(by="score", ascending=False, inplace=True)
        df.reset_index(inplace=True)
        df.index = pd.Index(np.arange(1,df.shape[0]+1), name="n_try")

        if scaling:
            scaling_coef = proportion_positive * df.shape[0]/df["score"].sum()
            df["scaled_score"] = scaling_coef * df["score"]
            df["expected (score)"] = df["scaled_score"].cumsum()
        else:
            df["expected (score)"] = df["score"].cumsum()

        df["expected (random)"] = proportion_positive * df.index
        df["n_true"] = df["y_true"].cumsum()

        return df


    def show_expected_value(self,
                            proportion_positive:float=None,
                            scaling:bool=True,
                            title:str="Expected value of number of positive instance",
                            xlabel:str="Number of tries",
                            ylabel:str="Number of positive instances",
                            expected_random:str="random",
                            expected_score:str="following score",
                            n_true:str="actual"
                            ):
        """

        :param proportion_positive:
        :param scaling:
        :param title:
        :param xlabel:
        :param ylabel:
        :param expected_random:
        :param expected_score:
        :param n_true:
        :return:
        """
        ## TODO: docstring
        df = self.optimize_expected_value(proportion_positive=proportion_positive,
                                          scaling=scaling)

        cols_orig = ["expected (random)", "expected (score)", "n_true"]
        cols_new = [expected_random, expected_score, n_true]

        for i,val in enumerate(cols_new):
            if not val:
                cols_new[i] = cols_orig[i]

        df_tmp = df[cols_orig].copy()
        df_tmp.columns = cols_new
        ax = df_tmp.plot()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


if __name__ == "__main__":
    pass
