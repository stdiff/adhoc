"""
Class for checking data quality
"""

from typing import Any, List, Union, Dict
from enum import Enum
from pathlib import Path
from datetime import datetime
from pytz import utc
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelBinarizer

from pylab import rcParams

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class NotInspectedError(Exception):
    pass


class VariableType(Enum):
    constant = "constant"
    binary = "binary"
    categorical = "categorical"
    continuous = "continuous"


def file_info(file: Union[str, Path]) -> pd.DataFrame:
    """
    Check the basic information about the file

    :param file: path to the file
    :return: DataFrame in a long format
    """
    file = Path(file)

    data = []
    data.append(("name", file.name))
    data.append(("size_in_MB", file.stat().st_size / (1024 * 1024)))

    with file.open("rb") as fo:
        md5sum = hashlib.md5(fo.read()).hexdigest()
        data.append(("md5sum", md5sum))

    return pd.DataFrame(data, columns=["key", "value"])


class Inspector:
    def __init__(self, df: pd.DataFrame, m_cats: int = 20):
        """
        Construct an inspection DataFrame of the given one
        Note that missing values are ignored for n_unique

        :param df: DataFrame to analyze
        :param m_cats: maximum number of values of a categorical variable
        """
        self.data = df  ## Do not take a copy. A reference is better.
        self._m_cats = m_cats
        self.result = None
        ## TODO: implementation of "string truncation of sampled values" (?)
        self.make_an_inspection()

    def make_an_inspection(self):
        self.result = pd.DataFrame(self.data.dtypes, columns=["dtype"]).astype(str)
        self.result["count_na"] = self.data.isna().sum()
        self.result["rate_na"] = self.data.isna().mean()
        self.result["n_unique"] = self.data.apply(lambda x: len(x.dropna(how="any").unique()), axis=0)
        self.result["distinct"] = self.result["n_unique"] == self.data.shape[0]
        self.m_cats = self._m_cats
        self.result["sample_value"] = self.data.apply(self.sample_value, axis=0)
        return self

    def detect_type(self, field: str) -> str:
        """
        find the variable type by using the result of inspection

        :param field: col name
        :return: "constant", "binary", "categorical" or "continuous"
        """
        if self.result is None:
            raise NotInspectedError

        s = self.result.loc[field, :]

        if s["n_unique"] <= 1:
            return VariableType.constant.name
        elif s["n_unique"] == 2:
            return VariableType.binary.name
        elif str(s["dtype"]).startswith("datetime"):
            return VariableType.continuous.name
        elif s["dtype"] == "object" or s["n_unique"] <= self._m_cats:
            return VariableType.categorical.name
        else:
            return VariableType.continuous.name

    @property
    def m_cats(self):
        return self._m_cats

    @m_cats.setter
    def m_cats(self, m_cats: int = 20):
        """
        The threshold of the categorical and continuous variables for numerical variable.
        (A variable of "object" is always categorical.)

        Detect the type of fields.

        binary: takes only two values. (This is also categorical.)
        categorical: is a object type or takes less than or equal to m_cats values
        continuous: takes more than m_cats values or datetime

        Remark:
        - We ignore NA to count the number of possible values of a field.
        - A categorical variable can be nominal (not ordered) or ordered.
        - This method detects all fields.

        :param m_cats: maximum number of values of a categorical variable
        :return: self
        """
        self._m_cats = m_cats
        self.result["variable"] = [self.detect_type(c) for c in self.result.index]

    def sample_value(self, s: pd.Series) -> Any:
        """
        return a non-missing value of the column in a random way.
        If the column has only missing values, then it will be returned.

        :param s: a value will be sampled from this given Pandas Series
        :return: a value
        """
        s = s.dropna().unique()

        if len(s):
            return np.random.choice(s, 1)[0]
        else:
            return np.nan

    def regard_as_categorical(self, field: str):
        """
        Modify of the variable type of the given field to categorical

        :param field: column name
        """
        ## TODO: accept a list of variable

        if field in self.result.index:
            if self.result.loc[field, "n_unique"] == 1:
                var_type = VariableType.constant

            elif self.result.loc[field, "n_unique"] == 2:
                var_type = VariableType.binary

            else:
                var_type = VariableType.categorical

            self.result.loc[field, "variable"] = var_type.name
        else:
            raise ValueError("No such field in the data set")

    def regard_as_continuous(self, field: str):
        """
        Modify of the variable type of the given field to continuous

        :param field: column name
        """
        ## TODO: accept a list of variable

        if field in self.result.index:
            self.result.loc[field, "variable"] = VariableType.continuous.name
        else:
            raise ValueError("No such field in the data set")

    def __repr__(self):
        return self.result.__repr__()

    def _repr_html_(self):
        return self.result._repr_html_()

    def get_cats(self) -> List[str]:
        """
        return the list of categorical variables

        :return: the list of categorical variables (including binary variables)
        """

        picked_up = [VariableType.binary.name, VariableType.categorical.name]
        df_cats = self.result.query("variable in @picked_up")
        return list(df_cats.index)

    def get_cons(self) -> List[str]:
        """
        return the list of continuous variables

        :return: the list of continuous variables
        """

        df_cons = self.result.query("variable == '%s'" % VariableType.continuous.name)
        return list(df_cons.index)

    def distribution_cats(self, fields: list = None, sort: bool = False) -> pd.DataFrame:
        """
        return a DataFrame showing the distribution of a variable.
        If no fields is specified, all categorical variable are checked.

        :param fields: list of (categorical) fields to check
        :param sort: if values should be sorted by count (ascending)
        :return: DataFrame of distributions
        """

        if fields is None:
            fields = self.get_cats()

        df_dist = []

        for field in fields:
            s = self.data[field]

            df_count = s.value_counts(dropna=True)

            if sort:
                df_count = df_count.sort_values(ascending=False).reset_index()
            else:
                df_count = df_count.sort_index().reset_index()

            df_count.columns = ["value", "count"]

            ## form indexes
            df_count["field"] = field
            df_count.set_index(["field", "value"], inplace=True)
            df_count["rate"] = df_count["count"] / len(s)

            df_dist.append(df_count)

        return pd.concat(df_dist, axis=0)

    def distribution_cons(self, fields: list = None):
        """
        return a DataFrame showing the distribution of the continuous variables.
        This is basically the same as df.describe().T

        :param fields: list of continuous fields to check
        :return: DataFrame of distributions
        """

        if fields is None:
            fields = self.get_cons()

        return self.data[fields].describe().T

    def distribution_timestamps(self, fields: List[str] = None):
        """
        return a DataFrame showing the distribution of the datetime variables.
        If no fields are given, use all datetime variables.

        :param fields: list of fields in datetime or date
        :return: DataFrame of distributions (looks like pandas.DataFrame.describe().T)
        """

        if fields is None:
            s_dtype = self.result["dtype"]
            is_datetime = s_dtype.apply(lambda x: x.startswith("datetime"))
            fields = s_dtype[is_datetime].index

        ## Compute the statistics by shifting by a base point.
        ## NB: This procedure is not necessary.

        df_stats = []
        for ts_col in fields:
            ts0 = self.data[ts_col].dropna().iloc[0]  ## pick a base point
            s_delta = self.data[ts_col] - ts0  ## shift by the base point

            if not isinstance(ts0, datetime):
                ## If the base point is not a datetime instance,
                ## then we can not shift them back by the base point,
                ## because of the data types. Therefore we convert it
                ## in a datetime format.
                ts0 = datetime(year=ts0.year, month=ts0.month, day=ts0.day, tzinfo=utc)

            ts_stats = pd.DataFrame(s_delta.describe()).T

            for idx in ts_stats.columns:
                if idx not in ["count", "std"]:
                    ## The standard deviation is a shift invariant,
                    ## therefore we do not shift it.
                    ts_stats[idx] += ts0

            df_stats.append(ts_stats)

        return pd.concat(df_stats, axis=0)

    ## Check if two variables are significantly different
    def significance_test(self, field1: str, field2: str, method: str = "spearman", verbose=True) -> pd.Series:
        """
        Execute a statistical test as follows
        - Both fields are categorical => chi-square test
        - Both fields are continuous => correlation
        - Otherwise => one-way ANOVA on ranks

        :param field1: field to compare
        :param field2: field to compare
        :param method: "spearman" (default) or "pearson"
        :param verbose: if warnings are shown
        :return: Series with index: field1, field2, test, statistic, pval
        """

        cats = self.get_cats()
        cons = self.get_cons()

        if field1 in cats and field2 in cats:
            #### chi2-test
            test = "chi-square test"
            contigency_table = pd.crosstab(self.data[field1], self.data[field2])

            if verbose and (contigency_table < 5).sum().sum() > 0:
                print("The contigency table (%s vs %s) contains too small cell(s)." % (field1, field2))
                print("Consult the documentation of stats.chi2_contingency")

            statistic, pval, dof, exp = stats.chi2_contingency(contigency_table)

        elif field1 in cons and field2 in cons:
            #### correlation
            if method == "spearman":
                test = "Spearman correlation"
                cor = stats.spearmanr
            else:
                test = "Peason correlation"
                cor = stats.pearsonr

            statistic, pval = cor(self.data[field1], self.data[field2])

        else:
            #### one-way ANOVA on ranks
            test = "one-way ANOVA on ranks"
            if field1 in cats and field2 in cons:
                cat, con = field1, field2
            elif field1 in cons and field2 in cats:
                cat, con = field2, field1
            else:
                raise ValueError("You gave a wrong field.")

            vals = self.data[cat].unique()

            samples = [self.data.loc[self.data[cat] == v, con] for v in vals]

            if verbose and any([len(s) < 5 for s in samples]):
                print("The groups withe less than 5 samples will be ignored.")
                samples = [x for x in samples if len(x) >= 5]

            statistic, pval = stats.kruskal(*samples)

        s = pd.Series([field1, field2, test, statistic, pval], index=["field1", "field2", "test", "statistic", "pval"])
        return s

    def significance_test_features(self, target: str, verbose: bool = False) -> pd.DataFrame:
        """
        Check the significance of feature variables against the target variables

        :param target: the target variable
        :param verbose: whether warnings are printed.
        :return: DataFrame containing the result of tests
        """
        fields = [f for f in self.data.columns if f != target]

        results = [self.significance_test(field, target, verbose=verbose) for field in fields]
        return pd.concat(results, axis=1).T

    def visualize_two_fields(self, field1: str, field2: str, heatmap: bool = False, rotation: float = 0.0):
        """
        Draw an informative diagramm for given two fields (feature and target).
        Note that this method can accept no constant field.

        :param field1: feature variable
        :param field2: target variable
        :param heatmap: a heatmap rather than a bar chart
        :param rotation: rotation of xticks
        """

        cats = self.get_cats()
        cons = self.get_cons()

        width, height = rcParams["figure.figsize"]
        aspect = width / height

        if field1 in cats and field2 in cats:
            ## bar chart or heat map
            df_ctab = pd.crosstab(self.data[field1], self.data[field2])

            df_ctab = df_ctab / df_ctab.sum(axis=0)  ## normalise

            if heatmap:
                sns.heatmap(df_ctab.T, annot=True, cmap="YlGnBu")
            else:
                df_ctab.plot.bar(stacked=False)

            title = "Distribution of %s by %s" % (field1, field2)

        elif field1 in cats and field2 in cons:
            ## violine
            sns.violinplot(field1, field2, data=self.data, inner="quartile")
            title = "Distribution of %s by %s" % (field2, field1)

        elif field1 in cons and field2 in cats:
            ## KDE
            sns.FacetGrid(self.data, hue=field2, height=height, aspect=aspect, legend_out=False).map(
                sns.kdeplot, field1, shade=True
            ).add_legend()
            plt.ylabel("density")
            title = "Kernel distribution estimate of %s by %s" % (field1, field2)

        elif field1 in cons and field2 in cons:
            ## joint plot
            sns.jointplot(field1, field2, data=self.data, kind="reg", height=height)
            title = ""

        else:
            raise ValueError("You gave a wrong field.")

        plt.xticks(rotation=rotation)
        plt.title(title)


class MultiConverter(BaseEstimator, TransformerMixin):
    def __init__(self, columns: list, cats: List[str], strategy: Dict[str, str] = None, drop: Dict[str, str] = None):
        """
        This Transformer applies `sklearn.impute.SimpleImputer`
        and `sklearn.preprocessing.LabelBinarizer`. Moreover
        we may remove one dummy variable for each categorical
        variables to avoid a collinear feature matrix.

        By definition, the keys of strategy is a subset of columns
        and the keys of drop must be a subset of cats.

        For strategy and drop you do not need to give all columns.
        Namely we apply the default behavior for not given columns.

        strategy: mean
        drop: most frequent values

        If you want to fill missing values with a constant value,
        then you can give it as "strategy". Then it will be considered
        as fill_value in SimpleImputer.

        If no dropping value is specified, we remove the most frequent
        value in the variable. "dropping value" for binary variable will
        be ignored because of the implementation of LabelEncoder.

        :param columns: list of column names
        :param cats: columns which LabelBinarizer is applied for
        :param strategy: column -> strategy for SimpleImputer
        :param drop: cat column -> value to drop
        """

        self.columns = columns
        self.strategy = strategy
        self.cats = cats
        self.drop = {} if drop is None else drop

        if not set(self.cats).issubset(self.columns):
            raise ValueError("cats must be subset of columns.")

        if not set(self.strategy.keys()).issubset(self.columns):
            raise ValueError("The keys of strategy must be subset of columns.")

        if not set(self.drop.keys()).issubset(self.cats):
            raise ValueError("The keys of drop must be a subset of cats")

        self.imputers = None
        self.lb = None
        self.classes_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        """
        Fit Imputers and LabelEncoders by taking into account the given information

        :param X: numpy array (2d) or pandas DataFrame
        :param y: no need
        :return: self
        """

        self.imputers = {}  ## col -> SimpleImputer
        self.lb = {}  ## col -> LabelBinarizer
        self.classes_ = []

        for i in range(X.shape[1]):
            ## fit SimpleImputer
            col = self.columns[i]

            if col in self.strategy.keys():
                strategy = self.strategy[col]
                if strategy not in ["mean", "median", "most_frequent"]:
                    fill_value = strategy
                    strategy = "constant"
                    imputer = SimpleImputer(strategy=strategy, fill_value=fill_value)
                else:
                    imputer = SimpleImputer(strategy=strategy)

            else:
                imputer = SimpleImputer(strategy="mean")

            if isinstance(X, pd.DataFrame):
                df_vals = X[[col]].copy()
            else:
                df_vals = pd.DataFrame({col: X[:, i]})

            df_vals.dropna(how="any", inplace=True)
            imputer.fit(df_vals)
            self.imputers[col] = imputer

            ## fit LabelEncoder
            if col in self.cats:

                values = list(df_vals[col].dropna().unique())
                lb = LabelBinarizer()
                lb.fit(values)
                self.lb[col] = lb

                ## dropping_value check
                if lb.y_type_ == "binary":
                    ## Because of the implementation of LabelEncoder we can not choose
                    ## the positive class.
                    dropping_value = lb.classes_[0]
                    self.drop[col] = dropping_value

                else:
                    if col in self.drop.keys():
                        dropping_value = self.drop[col]
                        if dropping_value not in values:
                            raise ValueError("%s is not found in the column %s" % (dropping_value, col))

                    else:
                        ## pick most frequent value
                        dropping_value = df_vals[col].value_counts().sort_values(ascending=False).index[0]
                        self.drop[col] = dropping_value

                classes = ["%s_%s" % (col, val) for val in lb.classes_ if val != dropping_value]
                self.classes_.extend(classes)

            else:
                self.classes_.append(col)

        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Convert the given numpy array or pandas DataFrame by applying trained
        imputers and encoders.

        :param X: numpy array (2d) or pandas DataFrame
        :return: numpy array
        """

        data = []
        for i in range(X.shape[1]):
            col = self.columns[i]

            if isinstance(X, pd.DataFrame):
                df_vals = X[[col]].copy()
            else:
                df_vals = pd.DataFrame({col: X[:, i]})

            ## filling missing values
            vals = self.imputers[col].transform(df_vals)  # ndarray of shape (n,1)

            ## one-hot encoding
            if col in self.lb.keys():
                vals = self.lb[col].transform(vals)  ## ndarray

                ## if the variable is not binary, then we drop the specified dummy variable
                if self.lb[col].y_type_ != "binary":
                    dropping_idx = list(self.lb[col].classes_).index(self.drop[col])
                    vals = np.delete(vals, obj=dropping_idx, axis=1)

            data.append(vals)

        return np.concatenate(data, axis=1)

    def inverse_transform(self, X):
        ## TODO: implementation inverse_transform
        raise NotImplemented("comming soon?")
