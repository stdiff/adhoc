"""
Class for checking data quality
"""

from typing import Any, List
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

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


class Inspector:

    def __init__(self, df:pd.DataFrame, m_cats:int=20):
        """
        Construct an inspection DataFrame of the given one
        Note that missing values are ignored for n_unique

        :param df: DataFrame to analyze
        :param m_cats: maximum number of values of a categorical variable
        """
        self.data = df ## Do not take a copy. A reference is better.
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


    def detect_type(self, field:str) -> str:
        """
        find the variable type by using the result of inspection

        :param field: col name
        :return: "constant", "binary", "categorical" or "continuous"
        """
        if self.result is None:
            raise NotInspectedError

        s = self.result.loc[field,:]

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
    def m_cats(self, m_cats:int=20):
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


    def sample_value(self, s:pd.Series) -> Any:
        """
        return a non-missing value of the column in a random way.
        If the column has only missing values, then it will be returned.

        :param s: a value will be sampled from this given Pandas Series
        :return: a value
        """
        s = s.dropna().unique()

        if len(s):
            return np.random.choice(s,1)[0]
        else:
            return np.nan


    def regard_as_categorical(self, field:str):
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


    def regard_as_continuous(self, field:str):
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


    def distribution_cats(self, fields:list=None, sort:bool=False) -> pd.DataFrame:
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
            df_count.set_index(["field","value"], inplace=True)
            df_count["rate"] = df_count["count"]/len(s)

            df_dist.append(df_count)

        return pd.concat(df_dist, axis=0)


    def distribution_cons(self,fields:list=None):
        """
        return a DataFrame showing the distribution of the continuous variables.
        This is basically the same as df.describe().T

        :param fields: list of continuous fields to check
        :return: DataFrame of distributions
        """

        if fields is None:
            fields = self.get_cons()

        return self.data[fields].describe().T


    def distribution_timestamps(self, fields:list=None):
        """
        return a DataFrame showing the distribution of the datetime variables.
        If no fields are given, use all datetime variables.

        :param fields: list of fields in datetime
        :return: DataFrame of distributions
        """

        if fields is None:
            s_dtype = self.result["dtype"]
            is_datetime = s_dtype.apply(lambda x: x.startswith("datetime"))
            fields = s_dtype[is_datetime].index

        df_stats = []

        for ts_col in fields:
            ts0 = self.data[ts_col].dropna().iloc[0] ## pick base point
            s_delta = self.data[ts_col] - ts0
            ts_stats = pd.DataFrame(s_delta.describe()).T

            for idx in ts_stats.columns:
                if idx not in ["count", "std"]:
                    ts_stats[idx] += ts0

            df_stats.append(ts_stats)

        return pd.concat(df_stats, axis=0)


    ## Check if two variables are significantly different
    def significance_test(self, field1:str, field2:str, method:str="spearman",
                          verbose=True) -> pd.Series:
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
                print("The contigency table (%s vs %s) contains too small cell(s)." % (field1,field2))
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

        s = pd.Series([field1, field2, test, statistic, pval],
                      index=["field1", "field2", "test", "statistic", "pval"])
        return s


    def significance_test_features(self, target:str, verbose:bool=False) -> pd.DataFrame:
        """
        Check the significance of feature variables against the target variables

        :param target: the target variable
        :param verbose: whether warnings are printed.
        :return: DataFrame containing the result of tests
        """
        fields = [f for f in self.data.columns if f != target]

        results = [self.significance_test(field, target, verbose=verbose) for field in fields]
        return pd.concat(results, axis=1).T


    def visualize_two_fields(self, field1:str, field2:str, heatmap:bool=False,
                             rotation:float=0.0):
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

        width, height = rcParams['figure.figsize']
        aspect = width / height

        if field1 in cats and field2 in cats:
            ## bar chart or heat map
            df_ctab = pd.crosstab(self.data[field1], self.data[field2])

            df_ctab = df_ctab / df_ctab.sum(axis=0) ## normalise

            if heatmap:
                sns.heatmap(df_ctab.T, annot=True, cmap="YlGnBu")
            else:
                df_ctab.plot.bar(stacked=False)

            title = "Distribution of %s by %s" % (field1, field2)

        elif field1 in cats and field2 in cons:
            ## violine
            sns.violinplot(field1, field2, data=self.data,
                           inner="quartile")
            title = "Distribution of %s by %s" % (field2, field1)

        elif field1 in cons and field2 in cats:
            ## KDE
            sns.FacetGrid(self.data, hue=field2,
                          height=height, aspect=aspect, legend_out=False)\
               .map(sns.kdeplot, field1, shade=True).add_legend()
            plt.ylabel("density")
            title = "Kernel distribution estimate of %s by %s" % (field1,field2)

        elif field1 in cons and field2 in cons:
            ## joint plot
            sns.jointplot(field1, field2, data=self.data, kind="reg",
                          height=height)
            title = ""

        else:
            raise ValueError("You gave a wrong field.")

        plt.xticks(rotation=rotation)
        plt.title(title)
        plt.show()

