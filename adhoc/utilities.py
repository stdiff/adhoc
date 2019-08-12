"""
Helper functions
"""

import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.utils import Bunch
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def bunch2dataframe(bunch:Bunch, target:str=None) -> pd.DataFrame:
    """
    Create a DataFrame from a Bunch instance. The Bunch instance must
    have attributes data (numpy array), feature_names (iterable) and
    target (numpy array or list).

    :param bunch: Bunch instance
    :param target: name of the target variable.
    :return: DataFrame
    """

    target = "target" if target is None else target
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df[target] = bunch.target
    return df


def load_iris(target:str="species") -> pd.DataFrame:
    """
    Construct a DataFrame from sklearn.datasets.load_iris

    :param target: name of the target variable
    :return: DataFrame
    """

    iris = datasets.load_iris()
    df = bunch2dataframe(iris, target)
    df.columns = [c[:-5].replace(" ","_") for c in iris.feature_names] + [target]
    df["species"] = df["species"].apply(lambda i: iris.target_names[i])
    return df


def load_boston(target:str="PRICE"):
    """
    Construct a DataFrame from sklearn.datasets.load_boston

    :param target: name of the target variable
    :return: DataFrame
    """
    boston = datasets.load_boston()
    return bunch2dataframe(boston, target)


def load_diabetes(target:str="progression"):
    """
    Construct a DataFrame from sklearn.datasets.load_diabetes

    :param target: name of the target variable
    :return: DataFrame
    """

    diabetes = datasets.load_diabetes()
    return bunch2dataframe(diabetes, target)


def bins_by_tree(data:pd.DataFrame, field:str, target:str,
                 target_is_continuous:bool, n_bins:int=5,
                 n_points:int=200, precision:int=2, **kwargs) -> pd.Series:
    """
    bin the given field by looking at the target variable. More precisely
    we bin the field, so that the cost function (Gini index/RMSE) is optimized.
    For implementation we just train a decision tree model.

    :param data: DataFrame containing field and target
    :param field: column to bin. Must be continuous.
    :param target: column which is used to calculate cost function.
    :param target_is_continuous: True if target variable is continuous.
    :param n_bins: number of bins
    :param n_points: number of points of grid to check
    :param precision: parameter of pandas.cut
    :param kwargs: parameter for DecisionTreeRegressor/Classifier
    :return: binned Series
    """

    if target_is_continuous:
        tree = DecisionTreeRegressor(max_leaf_nodes=n_bins, **kwargs)
    else:
        tree = DecisionTreeClassifier(max_leaf_nodes=n_bins, **kwargs)

    ## TODO: cross-validation
    tree.fit(data[[field]], data[target])

    ## grid space whose points can be separators
    grid = np.linspace(data[field].min(), data[field].max(), num=n_points)
    grid = pd.DataFrame({field:grid})

    if target_is_continuous:
        fibers = pd.DataFrame(tree.predict(grid))
    else:
        fibers = pd.DataFrame(tree.predict_proba(grid))

    ## possible values of predictions
    leaves = fibers.drop_duplicates().copy()
    leaves["bin"] = np.arange(0,leaves.shape[0]) ## put bin numbers

    grid = pd.merge(pd.concat([grid,fibers], axis=1), leaves, how="left", on=list(fibers.columns))

    ## find "bins" (boundaries of bins)
    bins = list(grid.groupby("bin")[field].first())
    bins[0] = -np.inf
    bins.append(np.inf)

    return pd.cut(data[field], bins, precision=precision)


from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

def bins_heatmap(data:pd.DataFrame, row:str, col:str, x:str, y:str, target:str,
                 n_bins:int=5):
    ## we assume that the target variable is continuous
    rows = data[row].unique()
    cols = data[col].unique()
    vmin, vmax = data[target].min(), data[target].max()

    fig, axes = plt.subplots(nrows=len(rows), ncols=len(cols))
    ax = list(axes.flat)

    for pair, subax in zip(product(rows, cols), ax):
        i,j = pair
        loc = np.logical_and(data[row] == i, data[col] == j)
        if not loc.any():
            continue

        subset = data.loc[loc,:].copy()

        for field in [x,y]:
            subset[field] = bins_by_tree(subset, field=field, target=target,
                                         target_is_continuous=True, n_bins=n_bins)

        df_heatmap = subset.groupby([x,y])[target].mean().reset_index()
        df_heatmap = df_heatmap.pivot(index=x, columns=y, values=target)
        sns.heatmap(df_heatmap, vmin=vmin, vmax=vmax, center=0.5, annot=True, ax=subax)

    plt.show()


if __name__ == "__main__":
    # df = load_iris()
    # species = df["species"].unique()
    # df["petal_length_bin"] = bins_by_tree(data=df, field="petal_length",
    #                                       target="species",
    #                                       target_is_continuous=False, n_bins=5)
    # df["petal_width_bin"] = bins_by_tree(data=df, field="petal_width",
    #                                      target="species",
    #                                      target_is_continuous=False, n_bins=5)
    # df = pd.concat([df, pd.get_dummies(df["species"])], axis=1)
    #
    # dg = df.groupby(["petal_length_bin","petal_width_bin"])[species].mean()
    # dg["majority"] = dg.apply(lambda r: r.idxmax(), axis=1)

    df = load_iris()
    df["cat"] = pd.qcut(df["sepal_length"], q=3)
    bins_heatmap(df, row="cat", col="species", x="petal_width", y="petal_length", target="sepal_width")


    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # df_bin = load_iris()
    # fields = ["petal_length", "petal_width"]
    # for field in fields:
    #     df_bin[field] = bins_by_tree(df_bin, field=field, target="sepal_width",
    #                                  target_is_continuous=True, n_bins=5)
    # df_bin = df_bin.groupby(fields)["sepal_width"].mean().reset_index()
    # df_bin = df_bin.pivot(index=fields[0], columns=fields[1], values="sepal_width")
    # sns.heatmap(df_bin, center=0.5, annot=True)
    # plt.show()

    #print(dg)
    #dg.reset_index(inplace=True)
    #dh = dg.pivot(index="petal_length_bin", columns="petal_width_bin", values="majority")
    #print(dh)

    #row = dg.iloc[0,:]
    #print(row)
