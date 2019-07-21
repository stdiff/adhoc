# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.3'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Usage of processing
#
# Please read [README.md](../README.md) in advance.

# ## Data Processing
#
# Data Processing means here a conversion from a (tabular) raw data to a feature matrix. This step contains **at least** following tasks.
#
# 1. check the quality of data
#    - How much is it filled? (NA string? missing rate?)
#    - Is the data valid? (Unexpected value in a column?)
#    - Is there anything strange/special?
# 2. understand the data
#    - What does each field look like?
#    - Is there any pair of two correlated fields?
#    - Is a feature variable informative for a prediction?
# 3. convert the given data into a suitable tabular format (feature matrix)
#    - How do we deal with missing values?
#    - How do we deal with date and time?
#    - What is a meaningful feature matrix for your prediction?
#
# Even though the analysis largely depends on the data set itself, many tasks can be modularized, so that we can avoid to modify your copy-and-pasted chunks to your data set.

# +
from datetime import datetime, timedelta, timezone
import pytz

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

# +
from IPython.display import display, HTML
plt.style.use("fivethirtyeight")

from pylab import rcParams
rcParams['figure.figsize'] = 14, 6 ## width, height (inches)

#pd.set_option('display.max_rows', None)
# -

import warnings
warnings.filterwarnings("ignore")

# As a sample dataset we use [Adult dataset](http://archive.ics.uci.edu/ml/datasets/Adult). You should consult the linked page for a brief explanation about the fields. The following unittest code downloads the data file and stores it as `../data/adult.csv`.

# %run ../test/test_data.py

utc = pytz.timezone("UTC")

# +
df = pd.read_csv("../data/adult.csv")

## dummy variable in datetime 
np.random.seed(51)
base_day = datetime(year=2019, month=4, day=1, tzinfo=utc)
df["dummy_ts"] = [base_day + timedelta(days=d) for d in 
                  np.random.normal(loc=0, scale=30, size=df.shape[0])]
df["dummy_ts"][0] = np.nan
df["dummy_ym"] = df["dummy_ts"].apply(lambda ts: ts.date().replace(day=1))

df.head()

# +
import sys
sys.path.append("..")

from adhoc.processing import Inspector
# -

# ### 1. Check the quality of data
#
# Creating an instance of `Inspector`, you can get an overview of the data quality of your dataset.

inspector = Inspector(df, m_cats=20)
inspector

# First of all the instance `inspector` is **not a DataFrame**. The default representation of the instance is the result of the inspection of the given DataFrame. You can access the DataFrame by the property `inspector.result`.

inspector.result.query("count_na > 0")

# #### Description of fields of `inspector.result`
#
# - dtype: This is the result of `df.dtypes`
# - count_na: The number of missing values (NA) in the column. `df.isna().sum()`
# - rate_na: The number of missing values (NA) in the column. `df.isna().mean()`
# - n_unique: The number of distinct values in the column. **We ignore missing values here.**
# - distinct: If a different row has a different number, then `True` else `False`. When it is `True`, then the column can be an ID such as a primary key or just a continuous variable.
# - variable: See below
# - sample_value: a randomly picked value. Note that we sample a value for all columns, not a row.
#
# As a basic policy we do not regard a missing value as a valid one, even though the missing value has a special meaning. This policy affects detection of variable types.
#
# #### Types of variables
#
# There are four kinds of types of variables.
#
# 1. `constant`: `n_unique==1`.
# 2. `binary`: it can take only two values. `n_unique==2`
# 3. `categorical`: it can take finite number of values. `dtype = "object"` or `n_unique <= m_cats`
# 4. `continuous`: it can take values in real numbers or timestamps.
#
# Remarks:
#
# - We do not care if there is an ordering of values: A nominal variable and an ordinal variable are both just a categorical variable. 
# - We regard a binary variable as a special case of a categorical variable.
# - A constant variable is neither categorical nor continuous.
# - The data type `datetime` is always regarded as a continuous variable, even though it can actually be a categorical variable (such as year-month).
#
# If you want to change the result, you can modify `m_cats` or use `regard_as_categorical()` or `regard_as_continuous()`.

inspector.regard_as_categorical("age")
inspector.result.loc["age",:] ## The variable type of age is categorical

# If we assign a number to `m_cats`, then the inspection is computed again. As a result your manual modification of variable types will be lost.

inspector.m_cats = 20
inspector.result.query("dtype == 'int64'") ## The variable type of age is now continuous.

# If you want to calculate the inspection once again because you converted a column, then `make_an_inspection()` does the job.

inspector.make_an_inspection()

# We can get easily the list of categorical/continuous variables. (Note that a constant variable is neither categorical nor continuous.)

print(inspector.get_cats()) ## list of categorical variables (binary or categorical)

print(inspector.get_cons()) ## list of continuous variables 

# #### Distributions 
#
# In order to find a special/strange values we need to check the distributions of the variables. 
#
# The distributions of categorical variables is shown by `distribution_cats()`

inspector.distribution_cats(fields=["workclass","dummy_ym"], sort=True)

# If `fields` is not given, then the distributions of all categorical variables are shown.
#
# You can also give selected fields to see their distribution in the above form. You can also give a continuous variable.

## Obviously a histogram is better and easier than this, but this is just an example.
inspector.distribution_cats(["age"]).reset_index(level=0)["count"].plot()
plt.xlabel("age")
plt.ylabel("count")
plt.title("The number of people by age");

# On the other hand, `distribution_cons()` shows just the same information as `df[inspector.get_cons()].describe().T`.

inspector.distribution_cons()

# Note that there is no statistics about `dummy_ts`. This is because the units of statistics are not unique. (e.g. `min` is time and `std` is duration.) `distribution_timestamps()` computes the statistics for such columns.

inspector.distribution_timestamps()

# Since `date` is recognized as an "object", `distribution_timestamps()` ignores `dummy_ym`, but you can compute the same statistics by giving the column name explicitly.

inspector.distribution_timestamps(fields=["dummy_ym"])

# ## 2. Understand the data
#
# It is most important to understand the meaning of each field. This must be the first step before analyzing the data. But it is not always the case that you can get a useful information about the fields. For examples the following explanation come from the data set description.
#
# > education: Bachelors, Some-college, 11th, HS-grad, Prof-school, Assoc-acdm, Assoc-voc, 9th, 7th-8th, 12th, Masters, 1st-4th, 10th, Doctorate, 5th-6th, Preschool.
# > 
# > education-num: continuous.
#
# What is `education-num`? What is the relation between them? The answer can be easily found by looing at the crosstab of two fields.

pd.crosstab(df["education"],df["education-num"])

# Namely `education-num` is just an alias of `education`. So we should remove one of them. In other words,
#
# 1. delete `education-num` and use `education` as a nominal variable. Or
# 2. delete `education` and use `education-num` as an ordinal variable.
#
# The default option would be 1, because the education level is not (linearly) "ordered".

# Next, what is the relation between `capital-gain` and `capital-loss`? Let us see the scatter plot of these two variables.

df.plot.scatter(x="capital-gain",y="capital-loss");

# It is easy to guess that one of the variables takes nonzero, then another is zero. 

(df["capital-gain"]*df["capital-loss"]).abs().sum() ## abs() is actually redundant.

# So why do we have two fields? They must be merged in one field, mustn't they?
#
# Or there must be a (relatively) clear relation between `workclass` and `label`. Looking at the crosstab of the two fields, we can find that `Never-worked` and `Without-pay` imply "low income". This is obvious, but important to check.

pd.crosstab(df["workclass"],df["label"])

# Anyway, it is very important to see the relation between two fields, because it is part of understanding data.
#
# |feature\target|categorical|continuous|
# |-|-|-|
# |categorical|Bar chart or heatmap|violine|
# |continuous|KDE or histogram|joint plot|

# #### categorical vs categorical
#
# The following bar chart shows the distributions of workclass by label. That is the sum of the (length/area of) blue/red bar is 1.

inspector.visualize_two_fields("workclass","label")

# We can show the same information by using heatmap. Since we compute the distribution of the `workclass` by `label`, the sum of each row in the heat map is 1.

inspector.visualize_two_fields("workclass","label", heatmap=True, rotation=10)

# If one or both categorical fields can take lots of values, then it is better to use just a crosstab or its heatmap. A visualization is not always the best solution.

inspector.visualize_two_fields("workclass","occupation")

inspector.visualize_two_fields("workclass","occupation", heatmap=True, rotation=10)

# #### categorical vs continuous
#
# The most popular visualization for this situation is probably a boxplot. But the whiskers are awkward to explain (to people who are not familiar with it).

# The violine plot does a similar job. This is just a visualisation of KDE. In my opinion, the violine plot is better than the box-and-whisker plot.

inspector.visualize_two_fields("marital-status", "age")

# #### continuous vs categorical
#
# This is quite similar to "categorical vs continuous". The difference is just the types of variables for the axes.
#
#
#
#
#

inspector.visualize_two_fields("age","label") ## con vs cat

# #### continuous vs continuous
#
# For the comparison between continuous variables `seaborn` provides a good visualization function: `pairplot`. This function also accepts an additional categorical variable as "hue". 

inspector.get_cons()

sns.pairplot(df, vars=inspector.get_cons()[:-1], hue="label");

# Here we do not have "dummy_ts" in the diagram again. The original reason is that `seaborn` (or `matplotlib`) does not support datetime in an expected way. (The following `jointplot` raises an exception and does not show histograms.)

try:
    sns.jointplot(x="dummy_ts", y="fnlwgt", data=df.dropna(how="any"))
except TypeError as e:
    print(e)
else:
    print("without error!")
finally:
    plt.xticks(rotation=90)

# If you want to draw time series (line plot), then `pandas.Series.plot` or `sns.tsplot` does the job.

x = df[["dummy_ts","fnlwgt"]].dropna(how="any").head(500).set_index("dummy_ts")["fnlwgt"]
x = x.sort_index()
x.plot();

# But it is more important to think about whether the scatter plot of (datetime, value) helps us understand data. Probably you need to aggregate the data by hour/day/month. 

# We should not forget a correlation heatmap. 

df.corr(method="spearman")

sns.heatmap(df.corr(method="spearman"), annot=True, linewidths=0.1, center=0, cmap="RdBu");

# Our module provides a short cut to the joint plot of two continuous variables.

inspector.visualize_two_fields("hours-per-week", "age", rotation=45)

# #### Statistical test 
#
# The diagramms are very helpful to understand the relation between fields and to get some insights. But It might be difficult to make a decision by looking only at diagramms: Is the field useful for making a prediction?
#
# In such a case we may execute statistical tests.
#
# |feature\target|categorical|continuous|
# |-|-|-|
# |categorical|$\chi^2$-test|ANOVA|
# |continuous|ANOVA|correlation|
#
# **Remark**
#
# 1. It is usual that the p-values you got are extremly small. In particular when you have a relatively large dataset.
# 2. The null-hypotheses of the tests are quite different.
#    - $\chi^2$-test : two categorical variables are independent.
#    - one-way ANOVA on ranks: consider the groups by the value of the categorical variable. Then the null-hypothesis is that the medians of the continuous variable in the groups are the same.
#    - Correlation: The two continuous variables are not correlated. 
# 3. The result of chi-square test for `dummy_ts` is shown. We have to think of whether it is meaningful.

inspector.significance_test_features("label", verbose=False)

# According to the above table the p-value of the ANOVA for `fnlwgt` and `label` is relatively large. In fact the KDE of `flnwgt` by `label` are quite similar.

inspector.visualize_two_fields("fnlwgt", "label")  ## con vs cat

# ## 3. Convert the DataFrame into a feature matrix
#
# 1. Fill missing values with the most-frequent value.
# 2. Apply a one-hot-encoder to each categorical variable.
#
# These two steps can be combined to one step by chosing some classes. If we just apply a one-hot-encoder to a categorical variable without missing values, then the feature matrix (including a constant column) has a colinear tuple of columns. The existance of linearly dependent variables is harmful when we train a linear model.

df["capital-gain"] = df["capital-gain"] - df["capital-loss"] ## merge the two fields

df.drop(["education-num","capital-loss","dummy_ym","dummy_ts"], axis=1, inplace=True)
inspector.make_an_inspection() ## update the inspection

# #### Fill missing values and one-hot encoding
#
# There are *at least* two steps to convert the DataFrame into a feature matrix:
#
# 1. Fill missing values
# 2. Convert a categorical variable into numerical column(s).
#
# ##### 1. Fill missing values
#
# First of all if a missing value has a special meaning, then we should fill them manually. But it is usual that missing values still remain after this step. In such a case we have to fill them by applying [a specific strategy](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html). 
#
# ##### 2. Convert a categorical variable into numerical column(s)
#
# We can not apply a machine learning model to a matrix with non numerical values. Most easiest way to achieve this is one-hot encoding. Assume that a categorical variable has $k$ classes. Then the original column (with $n$ instances) will be converted into an $n \times k$-matrix. Each column corresponds to a single class.
#
# An example is following. The first column is the "original" values and the rest 5 columns is the result of one-hot encoding.

# +
from sklearn.preprocessing import LabelBinarizer

lb = LabelBinarizer()
lb.fit(df["race"])

pd.concat([df["race"],
           pd.DataFrame(lb.transform(df["race"]), 
                        index=df.index, columns=lb.classes_)],
          axis=1).drop_duplicates()
# -

# The problem of one-hot encoding is that the feature matrix is always collinear you apply one-hot encoding to multiple columns. The collinearity causes a problem if we train a linear model: Training is very unstable. 
#
# Here is an illustrative example: A linear regression model predicting `age` by using two features `workclass` and `marital-status`. As you see the regression coefficients are too large/small. Even though the trained model can produce predictions in a reasonable range, the coefficients are not explainable.

# +
from sklearn.linear_model import LinearRegression

df_tmp = df[["workclass","marital-status","age"]].dropna(how="any")

lb0 = LabelBinarizer().fit(df_tmp["workclass"])
lb1 = LabelBinarizer().fit(df_tmp["marital-status"])

X = np.concatenate([lb0.transform(df_tmp["workclass"]),
                    lb1.transform(df_tmp["marital-status"])],
                    axis=1)
y = df_tmp["age"]

lr = LinearRegression()
lr.fit(X,y)
s_coef = pd.Series(lr.coef_, index=list(lb0.classes_)+list(lb1.classes_), name="coef")
s_coef["intercept"] = lr.intercept_
s_coef
# -

# If we drop one of the classes (a.k.a. dummy variable) for each categorical variable, then we obtain explainable coefficients:

# +
from sklearn.linear_model import LinearRegression

df_tmp = df[["workclass","marital-status","age"]].dropna(how="any")

lb0 = LabelBinarizer().fit(df_tmp["workclass"])
lb1 = LabelBinarizer().fit(df_tmp["marital-status"])

X = np.concatenate([lb0.transform(df_tmp["workclass"])[:,1:],
                    lb1.transform(df_tmp["marital-status"])[:,1:]],
                    axis=1)
y = df_tmp["age"]

lr = LinearRegression()
lr.fit(X,y)
s_coef = pd.Series(lr.coef_, index=list(lb0.classes_)[1:]+list(lb1.classes_)[1:], name="coef")
s_coef["intercept"] = lr.intercept_
s_coef
# -

# Theoretically no information will be lost, even though one of the classes is dropped.
#
# Class `MultiConverter` does the two steps at the same time. 

# +
from adhoc.processing import MultiConverter

cats = inspector.get_cats() ## variables for which LabelBinarizer is applied.
strategy = {cat:"most_frequent" for cat in cats} ## 

converter = MultiConverter(columns=df.columns, cats=cats, strategy=strategy)
converter.fit(df)

pd.DataFrame(converter.transform(df), columns=converter.classes_).head()
# -

# Consult the docstring for the usage of the class. Here is an overview of this transformation.
#
# 1. Apply `SimpleImputer` for all columns. The strategy of the imputation can be given through the `strategy` argument. **You should give strategies for all categorical variables**, because the default value of `strategy` is `mean` and this is nonsense for categorical variables. See the previous code chunk.
# 2. If `strategy` is not `most_frequent`, `mean` or `median`, then we fill the missing values with the given value.
# 3. We apply one-hot encoder for all categorical variables. As default the dummy variable corresponding to the most frequent class will be removed. 
# 4. You can give manually a dropping class to `drop` argument. But this is ignored if the categorical variable is binary.
# 5. `converter.classes_` gives you the original variable name and the corresponding class names.
# 6. `fit_transform()` is available. But you should not include this instance in `sklearn.pipeline.Pipeline` unless you specifies dropping values for all categorical variable. This is because the dropping variable can change during cross-validation.

pd.DataFrame.to_*?

df_transformed = pd.DataFrame(converter.transform(df), columns=converter.classes_)
df_transformed.to_pickle("../data/feature_matrix.pkl")

# ## Environment

# %load_ext watermark
# %watermark -v -n -m -p numpy,scipy,sklearn,pandas,matplotlib,seaborn
