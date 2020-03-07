# -*- coding: utf-8 -*-
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

# # Usage of modeling
#
# Please read [README.md](../README.md) in advance.

# ## Modeling
#
# Modeling means here training of several models and evaluate the trained models.
#
# We use the feature matrix which is obtained in [usage-processing.ipynb](./usage-processing.ipynb). If you have not executed it yet, try once or execute the following command on your terminal.
#
#      $ jupyter nbconvert --execute usage-processing.ipynb 

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

# The variable `label_>50K` is the target variable.

df = pd.read_pickle("../data/feature_matrix.pkl")
target = "label_>50K"
df.describe()

# First of all we split the data set into a training set and a test set.

# +
## Train-Test splitting
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df.drop(target, axis=1), df[target],
                                                    test_size=0.4, random_state=3)
print("Size of training set:", X_train.shape[0])
print("Size of test set    :", X_test.shape[0])
# -

# The distribution of the target variable is as follows.

y_train.value_counts()

100*y_train.mean() ## percentage of the label ">50K".

# ## Cross-Validation and Pipeline
#
# We will chose hyperparameters by cross-validation (CV). CV can be done by using `sklearn.model_selection.GridSearchCV`.
#
# It is often said that a training set should be rescaled before training. This is because the rescaling often optimizes the training so that we obtain a better model. To do so we construct a `Pipeline` instance, which `GridSearchCV` can be applied to. ([Good tutorial](https://iaml.it/blog/optimizing-sklearn-pipelines)) 
#
# ### Remark 
#
# A preprocessing algorithm (rescaling, PCA, etc.) is often applied to the training dataset and then CV is applied to the the preprocessed data set. This should be avoided, because the training preprocessing involves some information from the validation data. As a result the training preprocessing can cause "overfitting". `Pipeline` solves this problem.
#
# Because of the same reason we should actually integrate `adhoc.preprocessing.MultiConverter` to `Pipeline`. Namely `MultiConverter` fills missing values by some statistics ("mean", "median" and "most frequent class"). But such statistics are hardly ever overfitted if we have enough data, and it is more important that the column names are fixed. 
#
# Imagine you drop the dummy variable "White" in the first CV and "Other" in the second CV because the majority class can change. Namely your feature matrices can have different features. This is very confusing. Therefore we apply `MultiConverter` before CV.
#
# If you specify all dropping values manually, then you can integrate your `MultiConverter` instance in `Pipeline` without any problem.

try:
    from adhoc.modeling import grid_params, simple_pipeline_cv
except ImportError:
    import sys
    sys.path.append("..")
    from adhoc.modeling import grid_params, simple_pipeline_cv

# Let us try to train a model and pick the best hyperparameters. `grid_params` is a dict of simple `grid_param` for several models. You can use it for a simple analysis.
#
# `simple_pipline_cv` creates a simple `Pipeline` instance which consists of a Transformer for preprocessing (such as `MinMaxScaler`) and an ordinary estimator instance and put it in `GridSearchCV`. 
#
# Remark: We use 2-fold cross-validation just only for the CI-pipeline.

# +
from sklearn.linear_model import LogisticRegression

plr = simple_pipeline_cv(name="plr", model=LogisticRegression(penalty="elasticnet", solver="saga"), 
                         param_grid=grid_params["LogisticRegression"], cv=2) ## GridSearchCV instance
plr.fit(X_train, y_train);
# -

# The result of the cross-validation is stored in `cv_results_` attribute.

pd.DataFrame(plr.cv_results_)

# We can compute the confident interval of the cross-validation scores with `cv_results_summary`.

# +
from adhoc.modeling import cv_results_summary

cv_results_summary(plr)
# -

# If we have a linear model, then `show_coefficients` shows the regression coefficients.
#
# If you have a Pipeline with a scaler, then the regression coefficients are the coefficients after the scaler. Therefore you can use the absolute values of regression coefficients as "feature importance" (depending on your whole pipeline). But on the other hand you can not interpret the coefficients in the original scales.

# +
from adhoc.modeling import show_coefficients

show_coefficients(plr, X_train.columns).sort_values(by=1.0, ascending=False)

# +
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

tree = GridSearchCV(DecisionTreeClassifier(random_state=3),
                    grid_params["DecisionTree"], cv=2)
tree.fit(X_train,y_train)
cv_results_summary(tree)
# -

# A simple wrapper function for `sklearn.tree.export_graphviz` is available: 

# +
from adhoc.modeling import show_tree

show_tree(tree, X_train.columns)
# -
# Here is the feature importance of the decision tree.

# +
from adhoc.modeling import show_feature_importance

s_fi_tree = show_feature_importance(tree, X_train.columns)
s_fi_tree[s_fi_tree > 0]

# +
from sklearn.ensemble import RandomForestClassifier

rf = GridSearchCV(RandomForestClassifier(random_state=3),
                  grid_params["RandomForest"], cv=2)
rf.fit(X_train,y_train)
cv_results_summary(rf)
# -

s_fi_rf = show_feature_importance(rf, X_train.columns).sort_values(ascending=False)
s_fi_rf[s_fi_rf>0.01]

# +
from xgboost.sklearn import XGBClassifier

xgb_params = {"n_estimators":[10,20], "learning_rate":[0.1,0.01]}
xgb = GridSearchCV(XGBClassifier(max_depth=10, random_state=51),
                   xgb_params, cv=2)
xgb.fit(X_train,y_train)
cv_results_summary(xgb)
# -

s_fi_xgb = show_feature_importance(xgb, X_train.columns).sort_values(ascending=False)
s_fi_xgb[s_fi_xgb>0.01]

# ## Understanding a trained model
#
# Let's look at predictions of a trained decision tree model. According to its feature importance we create 2 continuous variables and 2 discrete variables, using the following variables.
#
# - capital-gain
# - marital-status_Never-married
# - marital-status_Divorced
# - education_Bachelors
# - education_Masters
# - hours-per-week
#
# Namely we create a categorical variables `marital-status` and `education`. In other words, we apply something like `LebelBinarizer.inverse_transform` with limited values. `adhoc.modeling.recover_label` does the job.

# +
from adhoc.modeling import recover_label

field2columns = {"marital-status": ["marital-status_Never-married", "marital-status_Divorced"],
                 "education": ["education_Bachelors", "education_Masters"]}

df_train = X_train.copy()
recover_label(df_train, field2columns, sep="_", inplace=True)

yhat = tree.predict_proba(X_train)
df_train["prob"] = yhat[:,1]
df_train["pred"] = tree.predict(X_train)
df_train[target] = y_train
# -

# `adhoc.modeling.recover_label` create a column `marital-status` out of two columns `marital-status_Never-married` and `marital-status_Divorced`. The rule is as follows:
#
# - `marital-status` is `Never_married` if `marital-status_Never-married == 1`
# - `marital-status` is `Divorced` if `marital-status_Divorced == 1` 
# - Otherwise `marital-status` is `other`.
#
# We do the similar preprocessing to create `education`. The following table show the concrete transformation.

# +
cols = []
for field, columns in field2columns.items():
    cols.append(field)
    cols.extend(columns)

df_train[cols].drop_duplicates()
# -

# The following scatter plot shows the predictions of the decision tree on the training set. The color shows the predicted probabilities: The red points are predicted as positive and the blue points are predicted as negative.

# +
from adhoc.utilities import facet_grid_scatter_plot

facet_grid_scatter_plot(data=df_train, col="marital-status", row="education", 
                        x="capital-gain", y="hours-per-week", c="prob", 
                        cmap="RdBu_r", margin_titles=True)
# -

# While the scatter plot gives a good overview of predictions, it is quite difficult to evaluate quantitatively just by looking at the diagrams. Therefore we bins x and y variables and show the average of probabilities for each pair of bins as a heat map.
#
# Each of the following heat maps corresponds to a pair of values of `marital-status` and `education`. Note that the bins are different according to heat maps. The choice of bins are optimized by decision trees. The bins without any color/any number has no instances.

# +
from adhoc.utilities import bins_heatmap

bins_heatmap(df_train, cat1="marital-status", cat2="education", x="capital-gain", y="hours-per-week",
             target="prob", center=0.5, cmap="RdBu_r", fontsize=14)
# -

# Just in case. The above heat maps show the average probabilities which are predicted by a trained decision tree model on the training set, not the true values of the target variable.

# ## Evaluation of the model by AUC
#
# In some cases (especially when the target variable is a skewed binary variable) you need to evaluate of a trained model by a special metric such as AUC.
#
# **Remark**: If you want to do the following analysis, I strongly recommend to split your original dataset into 3 datasets: training set, validation set and test set. You train a model, choosing "roc_auc" as a scoring function and do the following analysis with the validation set.
#
# (Why we did not do so? Well, it might be confusing at glance. *Why don't we use the validation set for choosing hyperparameters?*)
#
# ### Lead prioritization
#
# Let us assume that our data set is a list of customers and the target customers of your product (or service) is people who earn relatively much. And therefore you have decided to contact customers directly (for example by telephone) instead of on-line advertising to huge number of audience. Because it is time-consuming to contact a customer directory, you want to reduce the number of contacts. 
#
# - A contacting a customer costs €40 on average.
# - If the customer buys your product, you get €1000. (This is the price.)
#
# (These numbers have no reality. Just an assumption.)
#
# Since we do not have a data set for successful customers, so we naïvely assume that 10% of the customers with label "`>50K`" buy your product. Because 24% of the customers have label "`>50K`", the proportion of the customer buying the product would be 2.4%. (Of course, your "test set" has no information about the label, neither.)
#
# If you randomly choose a customer and contact her/him. Then your success rate is exactly 2.4%. That is, you get a customer buying your product if you contact 42 customers on average. And therefore getting such a customer costs €1680 on average. This is a deficit. On the other hand if you can perfectly find a positive customer. Then your success rate will be 10%. That is, you will get a customer buying your product by contacting 10 customers on average and it costs €400.
#
# One of the difficulties of this challenge is that the accuracy of the model is not a right metric. Let us look at the performance of the random forest classifier (on the training set).

# +
from adhoc.modeling import ROCCurve

def performance_check(model, X:pd.DataFrame, y:pd.Series, threshold:float=0.5) -> pd.DataFrame:
    roc_curve = ROCCurve(y,model.predict_proba(X)[:,1])
    score = roc_curve.get_scores_thru_threshold(threshold=threshold)
    ct = roc_curve.get_confusion_matrix(threshold=threshold)
    
    n_contact = ct.loc[1,:].sum()
    success_rate = 0.1*score["precision"]
    contact_cost = 40
    price = 1000
    profit = price*success_rate*n_contact - contact_cost*n_contact
    
    print("- Your model has %d%% accuracy." % (100*score["accuracy"]))
    print("- You will contact %d customers." % n_contact)
    print("- You can reach %d%% of positive customers" % (100*score["recall"]))
    print("- Your success rate is %0.1f%%" % (100*success_rate))
    print("- Your profit would be %d" % profit)
    return ct

performance_check(rf, X_train, y_train)
# -

# Let's look at another model: XGBoosting.

performance_check(xgb, X_train, y_train)

# So what is the right metric?
#
# 1. Profit. Then you definitely want to choose the XGBoosting model. 
# 2. Success rate (or precision). If you want to find a small number of customers buying your product quickly, because it is challenging and therefore you want to receive feedback quickly to improve your product and to release a new version. Of course the assumption that 10% of positive customers buy the product can be too optimistic, so you want to change strategy in an early stage.
#
# If you choose the second option, then you want to choose the random forest classifier because of the high success rate. But you can obtain a better model by tweaking threshold of the XGBoosting model:

performance_check(xgb, X_train, y_train, threshold=0.8)

# The classifiers which we trained predict actually probabilities that a customer is positive and therefore we naturally use 0.5 as a threshold/boundary of predictions: If the probability is larger than 0.5, then the customer would be positive. So why don't we start with the customer with high probabilities? In the third crosstab we use 0.8 as the threshold, then you will achieve a model with a better precision. Here we should note that the accuracy becomes worse than the model with threshold 0.5.
#
# In general, the more accurate the classifier is, the better the predictions are. But this is not always the case, especially when you have a specific metric.
#
# There is a problem. The precision is often not a good metric to optimize. You can also achieve that a random forest classifier with a good precision just by choosing a high threshold:

performance_check(rf, X_train, y_train, threshold=0.7)

# But as you see, you can reach only 338 positive customers. Is it enough to close 33 contracts? If the assumption is too optimistic, you can find less than 33 customers buying your product. That is, it is also important to reach larger number of positive customers. Therefore you have actually two measures to optimize.
#
# - Precision: the proportion of positive customers among the customer you will contact.
# - Recall: the proportion of positive customers you can reach.
#
# There are two problems:
#
# 1. The metrics are determined after choosing a threshold. Then how should we train a model and tune hyperparameters?
# 2. It is problematic that there are two metrics to optimize. Which has priority and how do we measure it.
#
# The standard solutions to the above questions are following:
#
# 1. Train a model in a usual way, but tune hyperparameters by looking at area under the ROC curve.
# 2. Use F1-metric and choose the threshold which maximizes it. Or optimize your metric by varying threshold.
#
# (You might want to perform a special sampling method if you need.)
#
# ### ROC curve and AUC
#
# Assume that you have a trained model predicting probabilities (or scores). Then choosing a threshold, you have a crosstab as above, and therefore you also have False-Positive rate (FP rate) and True-Positive rate (TP rate) of the predictions. The ROC (Receiver Operating Characteristic) curve is the curve of pairs (FP rate, TP rate) by varying the threshold.
#
# The following line curve is the ROC curve of the XGBoosting model.

y_score_xgb = xgb.predict_proba(X_train)[:,1]
roc_curve_xgb = ROCCurve(y_true=y_train, y_score=y_score_xgb)
roc_curve_xgb.show_roc_curve()

# Usually this curve moves from (1,1) (Every customer is positive) to (0,0) (Every customer is negative) and "usually" the curve lies over the diagonal line (dotted line in the diagram). (If not, you have a wrong classifier.) The ROC curve shows the performance of your model with all possible threshold.
#
# Since the upper left point (FP=0, TP=1) corresponds the perfect classifier, the ROC curve approaches upper left corner if your model can predict correctly. We can measure "how the ROC curve approaches to upper left" as the area under the ROC curve (AUC).
#
# The following curve is the ROC curve of the random forest classifier. AUC is slightly worse than one of XGBoosting.

y_score_rf = rf.predict_proba(X_train)[:,1]
roc_curve_rf = ROCCurve(y_true=y_train, y_score=y_score_rf)
roc_curve_rf.show_roc_curve()

# Let's look at the F1 scores. We recall that the F1-score is defined by the harmonic mean of $P$ and $R$:
#
# $$F = \dfrac{2PR}{P+R}$$
#
# Here $P$ is the precision and $R$ is the recall. The following heat map shows the F1-scores for various precisions and recalls.

# +
from itertools import product

vals = np.round(0.1*np.arange(11),1)
df_f1 = pd.DataFrame(list(product(vals,vals)), columns=["precision","recall"])
df_f1["F1-score"] = 2*df_f1["precision"]*df_f1["recall"]/(df_f1["precision"]+df_f1["recall"])
df_f1 = df_f1.pivot(index="recall", columns="precision", values="F1-score").sort_index(ascending=False)
sns.heatmap(df_f1, cmap="YlGnBu", annot=True)
plt.title("Table of F1-scores");
# -

# Let us look at line curves of the metrics (recall, precision and F1-score) by thresholds.

roc_curve_xgb.show_metrics() ## takes relatively long

# The precise values of metrics can be obtain by `scores` property:

roc_curve_xgb.scores.head()

# By using this it is easy to find the threshold which maximizes F1-score.

roc_curve_xgb.scores.sort_values(by="f1_score", ascending=False).head()

best_threshold = roc_curve_xgb.scores.sort_values(by="f1_score", ascending=False).index[0]
roc_curve_xgb.get_confusion_matrix(threshold=best_threshold)

# The above crosstab is the result of the threshold with the best F1-score. Of course you can choose another threshold, for example 0.7, so that you obtain a better precision.

roc_curve_xgb.get_confusion_matrix(threshold=0.7)

# Or you might want to choose the threshold which maximizes the profit.

# +
df_performance = roc_curve_xgb.scores.copy()
price = 1000
cost = 40

df_performance["success_rate"] = 0.1*df_performance["precision"]
df_performance["n_contract"] = (df_performance["success_rate"]*df_performance["n_pos_pred"]).astype(int)
df_performance["profit"] = price*df_performance["n_contract"] - cost*df_performance["n_pos_pred"]
df_performance["profit"].plot()
plt.title("Profit by threshold");
# -

# The best threshold is 0.366727 and the profit (on training set) will be €160 560.

df_performance.sort_values(by="profit", ascending=False).head(1)

# Another usage of the trained model is to use probabilities as scores. The higher the score is, the more likely the customer is positive. Therefore we can contact positive customers by contacting customers in the descending order of scores.
#
# First of all the expected value of a number of positive customer agrees with the proportion of positive customers if we contact randomly. Now we have a probability of positivity for each customer. Then the expected value of the number of positive customers is just the sum of the probabilities.
#
# Here is a problem. What is the expected value of the whole (training) dataset?

print("Expected value (random contact)         : %s" % y_train.sum())
print("Expected value (predicted probabilities): %s" % y_score_xgb.sum())

# In other words, the model says that you can reas more 5000 positive customers. This can not happen. But this is normal because we do not train a model so that the expected values agree. One of the easiest solution to the problem is scaling. That is, we multiply a certain constant so that the expected value of the number of positive customers with the predicted probability agrees with the expected value with equally likely probabilities.
#
# Then the expected values of number of positive customers by the number of contacts look like following.

proportion_positive = y_train.mean()
roc_curve_xgb.show_expected_value(proportion_positive=proportion_positive, scaling=True)

# According to the graph you could reach 2721 positive customers if you contact 5000 customers in the descending order of scores. (But you will actually reach 3591 positive customers.) If you contact randomly, then you can reach only 1195 positive customer.

roc_curve_xgb.optimize_expected_value(proportion_positive=proportion_positive, scaling=True).loc[5000,:]

# +
def profit_df(roc:ROCCurve, price:int, cost:int) -> pd.DataFrame:
    df_profit = roc.optimize_expected_value(proportion_positive=proportion_positive, 
                                                  scaling=True).copy()
    df_profit["following score"] = price*0.1*df_profit["expected (score)"] - cost*df_profit.index
    df_profit["random"] = price*0.1*df_profit["expected (random)"] - cost*df_profit.index
    df_profit["actual"] = price*0.1*df_profit["n_true"] - cost*df_profit.index
    return df_profit


df_profit = profit_df(roc_curve_xgb, price, cost)
best_n_try = df_profit.sort_values(by="following score", ascending=False).index[0]

def show_profit(df_profit:pd.DataFrame, xintercept:int):
    cols = ["random", "following score", "actual"]
    df_profit[cols].plot()
    plt.vlines(xintercept, ymin=df_profit["actual"].min(), ymax=df_profit["actual"].max(),
               linestyle=":")
    plt.title("Expected profit by number of contacts")
    plt.xlabel("Number of contacts")
    plt.ylabel("Profit");
    
show_profit(df_profit, best_n_try)
# -

# According to the graph the profit is maximized if we contacts 3637 customers in the descending order of scores. The expected profit will be €78025. This number is very small than the profit we compute above. This is because of the scaling. 

df_profit.loc[best_n_try,:]

# We have seen two approaches:
#
# 1. Contact all customers which are predicted as positive customers.
# 2. Use predicted probabilities as scores and contact customers in the descending order of scores.
#
# **You should not show the both approaches to the audience.** They are definitely confused by the multiple solutions and you are going to be asked which solution is correct/reliable.
#
# You should make it clear how the predicted model is going to be used? What is the goal of the analysis? If it is clear, then you should choose only one metric to optimize.

# ### Evaluation of the model on test set

y_score = xgb.predict_proba(X_test)[:,1]
roc = ROCCurve(y_true=y_test, y_score=y_score)
roc.show_roc_curve()

# When computing expected value, we have to use the proportion of the positive customers in the **training set** (not the test set). This is because we do not know the actual proportion of positive customers in the test set and we also have to predict it.

roc.show_expected_value(proportion_positive=proportion_positive, scaling=True)

df_profit = profit_df(roc, price, cost)
best_n_try_test = df_profit.sort_values(by="following score", ascending=False).index[0]
show_profit(df_profit, best_n_try_test)

df_profit.loc[best_n_try_test,:]

# ## Environment

# %load_ext watermark
# %watermark -v -n -m -p numpy,scipy,sklearn,pandas,matplotlib,seaborn
