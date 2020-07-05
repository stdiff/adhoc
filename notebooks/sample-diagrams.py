# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Sample Diagrams
#
# The aim of this notebook is to provide sample codes creating diagrams.

import numpy as np
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# +
from IPython.display import display, HTML
plt.style.use("fivethirtyeight")

from pylab import rcParams
rcParams['figure.figsize'] = 14, 6
# -

try:
    from adhoc.processing import Inspector
    from adhoc.modeling import show_tree
    from adhoc.utilities import load_iris, facet_grid_scatter_plot, bins_heatmap
except ImportError:
    import sys
    sys.path.append("..")
    from adhoc.processing import Inspector
    from adhoc.modeling import show_tree
    from adhoc.utilities import load_iris, facet_grid_scatter_plot, bins_heatmap

# +
np.random.seed(1)

df = load_iris(target="species")
df["cat1"] = np.random.choice(["a","b","c"], size=df.shape[0], replace=True)
df["cat2"] = (df.iloc[:,0]*df.iloc[:,1] - df.iloc[:,2]*df.iloc[:,3] > 11).map({True:1,False:0})

inspector = Inspector(df)
inspector ## 4 continuous variables and 3 categorical variables
# -

inspector.visualize_two_fields("sepal_width","sepal_length") ## continuous x continuous

inspector.visualize_two_fields("petal_width","species") ## continuous x categorical

inspector.visualize_two_fields("species","petal_width") ## categorical x continuous

inspector.visualize_two_fields("species","cat2")

inspector.visualize_two_fields("species", "cat2", heatmap=True)

# +
from sklearn.tree import DecisionTreeClassifier
features = ["sepal_length","sepal_width","petal_length","petal_width","cat2"]

tree = DecisionTreeClassifier(max_depth=3, random_state=4)
tree.fit(df[features],df["species"]);
# -

show_tree(tree, columns=features)

facet_grid_scatter_plot(df, row="species", col="cat1", 
                        x="petal_width", y="petal_length", 
                        c="sepal_width", cmap="Greens")

facet_grid_scatter_plot(df, row="cat1", col="cat2", 
                        x="petal_width", y="petal_length", hue="species")

bins_heatmap(df, cat1="cat1", cat2="cat2", x="petal_width", y="petal_length",
             target="sepal_width", fontsize=14)

# ## Envirmediant

# %load_ext watermark
# %watermark -v -n -m -p numpy,scipy,sklearn,pandas,matplotlib,seaborn
