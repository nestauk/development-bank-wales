# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: development_bank_wales
#     language: python
#     name: development_bank_wales
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import asf_core_data
from asf_core_data.pipeline.preprocessing import (
    preprocess_epc_data,
    feature_engineering,
)
from asf_core_data.pipeline.data_joining import merge_install_dates
from asf_core_data.getters.epc import epc_data
from asf_core_data.getters.supplementary_data.deprivation import imd_data

from asf_core_data.utils.visualisation import easy_plotting, kepler
from asf_core_data.utils.geospatial import data_agglomeration

from development_bank_wales import PROJECT_DIR, Path
from development_bank_wales.pipeline import recommendations, upgrades

import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keplergl import KeplerGl
import seaborn as sns

from ipywidgets import interact, fixed


# %%
LOCAL_DATA_DIR = "/Users/juliasuter/Documents/ASF_data"

wales_eng_df = recommendations.load_epc_certs_and_recs(
    data_path=LOCAL_DATA_DIR,
    subset="Wales",
    n_samples=None,
    remove_duplicates=False,
    no_merge=True,
)

# %%
wales_eng_df = pd.read_csv(
    "/Users/juliasuter/Documents/ASF_data/outputs/EPC/preprocessed_data/2021_Q4_0721/EPC_Wales_preprocessed.csv"
)

# %%
# wales_eng_df = epc_data.load_preprocessed_epc_data(data_path=LOCAL_DATA_DIR,version='preprocessed')

# %%
wales_eng_df = wales_eng_df.loc[wales_eng_df["TENURE"] == "owner-occupied"]

# %%
wales_eng_df = merge_install_dates.manage_hp_install_dates(wales_eng_df)

# %%
wales_eng_df.to_csv(PROJECT_DIR / "outputs/temp_data.csv")

# %%
wales_eng_df = pd.read_csv(PROJECT_DIR / "outputs/temp_data.csv")

# %%
a = wales_eng_df.ROOF_DESCRIPTION.value_counts()
m = wales_eng_df.ROOF_DESCRIPTION.isin(a.index[a < 1000])
wales_eng_df.loc[m, "ROOF_DESCRIPTION"] = np.NaN

# %%
pd.set_option("display.max_rows", 1000)
wales_eng_df["ROOF_DESCRIPTION"].value_counts()

# %%
# wales_eng_df = wales_eng_df.loc[wales_eng_df['TENURE'] == 'owner-occupied']

latest_wales = epc_data.filter_by_year(wales_eng_df, None, selection="latest entry")
first_wales = epc_data.filter_by_year(wales_eng_df, None, selection="first entry")

upgrade_df = upgrades.get_upgrade_features(first_wales, latest_wales, keep="first")

# %%
upgrade_df["ROOF_EFF_DIFF"].value_counts(normalize=True) * 100

# %%
upgrade_df["ROOF_DESCRIPTION"].value_counts(normalize=True) * 100

# %%
features = upgrade_df
label = "ROOF_EFF_DIFF"

# %%
features.to_csv(PROJECT_DIR / "outputs/features.csv")

# %%
label = "ROOF_EFF_DIFF"

# %%
features = pd.read_csv(PROJECT_DIR / "outputs/features.csv").drop(
    columns=["Unnamed: 0.1", "Unnamed: 0"]
)

# %%
list(features.columns)

# %%
drop_features = [
    "LMK_KEY",
    "ADDRESS1",
    "ADDRESS2",
    "POSTCODE",
    "BUILDING_REFERENCE_NUMBER",
    "LODGEMENT_DATE",
    "LODGEMENT_DATE",
    "LIGHTING_COST_CURRENT",
    "LIGHTING_COST_POTENTIAL",
    "HEATING_COST_CURRENT",
    "HEATING_COST_POTENTIAL",
    "HOT_WATER_COST_CURRENT",
    "HOT_WATER_COST_POTENTIAL",
    "MULTI_GLAZE_PROPORTION",
    "GLAZED_TYPE",
    "GLAZED_AREA",
    "HOT_WATER_DESCRIPTION",
    "HOT_WATER_ENERGY_EFF",
    "FLOOR_DESCRIPTION",
    "FLOOR_ENERGY_EFF",
    "WINDOWS_DESCRIPTION",
    "WINDOWS_ENERGY_EFF",
    "SECONDHEAT_DESCRIPTION",
    "MAINHEAT_DESCRIPTION",
    "MAINHEAT_ENERGY_EFF",
    "MAINHEATC_ENERGY_EFF",
    "LIGHTING_DESCRIPTION",
    "LIGHTING_ENERGY_EFF",
    "MAIN_FUEL",
    "UPRN",
    "HOT_WATER_ENERGY_EFF_SCORE",
    "FLOOR_ENERGY_EFF_SCORE",
    "WINDOWS_ENERGY_EFF_SCORE",
    "MAINHEAT_ENERGY_EFF_SCORE",
    "MAINHEATC_ENERGY_EFF_SCORE",
    "LIGHTING_ENERGY_EFF_SCORE",
    "BUILDING_ADDRESS_ID",
    "HEATING_SYSTEM",
    "HEATING_FUEL",
    "HP_INSTALLED",
    "HP_TYPE",
    "CURR_ENERGY_RATING_NUM",
    "ENERGY_RATING_CAT",
    "original_address",
    "HP_INSTALL_DATE",
    "MCS address",
    "FIRST_HP_MENTION",
    "ANY_HP",
    "HP_AT_FIRST",
    "HP_AT_LAST",
    "HP_LOST",
    "HP_ADDED",
    "HP_IN_THE_MIDDLE",
    "MCS_AVAILABLE",
    "HAS_HP_AT_SOME_POINT",
    "ARTIFICIALLY_DUPL",
    "EPC HP entry before MCS",
    "No EPC HP entry after MCS",
    "MECHANICAL_VENTILATION",
    "CHANGE_WALLS_DESCRIPTION",
    "UPGRADED_WALLS_DESCRIPTION",
    "UPGRADABLE_WALLS",
    "UPGRADABILITY_WALLS",
    "CHANGE_ROOF_DESCRIPTION",
    "UPGRADED_ROOF_DESCRIPTION",
    "UPGRADABLE_ROOF",
    "UPGRADABILITY_ROOF",
    "MAINHEAT_EFF_DIFF",
    "CHANGE_MAINHEAT_DESCRIPTION",
    "UPGRADED_MAINHEAT_DESCRIPTION",
    "UPGRADABLE_MAINHEAT",
    "UPGRADABILITY_MAINHEAT",
    "HOT_WATER_EFF_DIFF",
    "CHANGE_HOT_WATER_DESCRIPTION",
    "UPGRADED_HOT_WATER_DESCRIPTION",
    "UPGRADABLE_HOT_WATER",
    "UPGRADABILITY_HOT_WATER",
    "LIGHTING_EFF_DIFF",
    "CHANGE_LIGHTING_DESCRIPTION",
    "UPGRADED_LIGHTING_DESCRIPTION",
    "UPGRADABLE_LIGHTING",
    "UPGRADABILITY_LIGHTING",
    "FLOOR_EFF_DIFF",
    "CHANGE_FLOOR_DESCRIPTION",
    "UPGRADED_FLOOR_DESCRIPTION",
    "UPGRADABLE_FLOOR",
    "UPGRADABILITY_FLOOR",
    "WINDOWS_EFF_DIFF",
    "CHANGE_WINDOWS_DESCRIPTION",
    "UPGRADED_WINDOWS_DESCRIPTION",
    "UPGRADABLE_WINDOWS",
    "UPGRADABILITY_WINDOWS",
    "UPGRADABILITY_TOTAL",
    "TOTAL_EFF_DIFF",
    "LOCAL_AUTHORITY_LABEL",
    "TRANSACTION_TYPE",
    "WALLS_ENERGY_EFF",
    "ANY_UPGRADES",
]


# %%
features.drop(columns=drop_features, inplace=True)

# %%
features["ROOF_UPGRADABILITY"] = (features["ROOF_EFF_DIFF"] > 0.0).astype(int)
label = "ROOF_UPGRADABILITY"
features.drop(columns=["ROOF_EFF_DIFF"], inplace=True)

# %%
features["ROOF_UPGRADABILITY"].value_counts(normalize=True)

# %%
from development_bank_wales.pipeline.machine_learning import model_preparation

features = model_preparation.balance_set(
    features, label, false_ratio=0.75, binary=False
)
print(features.shape)

# %%
features["ROOF_UPGRADABILITY"].value_counts(normalize=True)

# %%
from development_bank_wales.pipeline.encoding import one_hot_encoding

features = one_hot_encoding.feature_encoding_pipeline(
    features
    #   drop_features=['ADDRESS1', 'ADDRESS2','INSPECTION_DATE','LOCAL_AUTHORITY_LABEL',
    #                  'BUILDING_ADDRESS_ID','BUILDING_REFERENCE_NUMBER',
    #                 'CHANGE_WALLS_DESCRIPTION', 'CHANGE_WALLS_DESCRIPTION','CHANGE_ROOF_DESCRIPTION',
    #                 'CHANGE_MAINHEAT_DESCRIPTION','CHANGE_LIGHTING_DESCRIPTION','CHANGE_FLOOR_DESCRIPTION',
    #                  'CHANGE_WINDOWS_DESCRIPTION', 'POSTCODE', 'LMK_KEY', 'UPRN'
)


features.dropna(axis=1, how="all", inplace=True)
nunique = features.nunique()
cols_to_drop = nunique[nunique == 1].index
features.drop(cols_to_drop, axis=1, inplace=True)


# %%
# Use numpy to convert to arrays
import numpy as np

# Labels are the values we want to predict
labels = np.array(features[label])  # Remove the labels from the features
# axis 1 refers to the columns
features = features.drop(label, axis=1)
# Saving feature names for later use
feature_list = list(features.columns)
# Convert to numpy array
features = np.array(features)
features.shape

# %%
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

prepr_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(missing_values=np.nan, strategy="median")),
        ("min_max_scaler", MinMaxScaler()),
        # ("pca", PCA(n_components=0.9, random_state=42)),
    ]
)

features = prepr_pipeline.fit_transform(features)
features.shape

# %%
import sklearn

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(
    features, labels, test_size=0.25, random_state=42, stratify=labels
)

# %%
print("Training Features Shape:", train_features.shape)
print("Training Labels Shape:", train_labels.shape)
print("Testing Features Shape:", test_features.shape)
print("Testing Labels Shape:", test_labels.shape)

# %%
# Import the model we are using
from sklearn.ensemble import RandomForestClassifier

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model on training data
rf.fit(train_features, train_labels)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

y_pred_test = rf.predict(test_features)
accuracy_score(test_labels, y_pred_test)

# %%
matrix = confusion_matrix(test_labels, y_pred_test)
matrix = matrix.astype("float") / matrix.sum(axis=1)[:, np.newaxis]

# Build the plot
plt.figure(figsize=(16, 7))
sns.set(font_scale=1.4)
sns.heatmap(
    matrix, annot=True, annot_kws={"size": 10}, cmap=plt.cm.Greens, linewidths=0.2
)

# Add labels to the plot
class_names = ["Not upgradable", "Upgradable"]
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25)
plt.yticks(tick_marks2, class_names, rotation=0)
plt.xlabel("Predicted label")
plt.ylabel("True label")
plt.title("Confusion Matrix for Random Forest Model")
plt.show()

# %%
print(classification_report(test_labels, y_pred_test))

# %%
# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
# Calculate the absolute errors
errors = abs(predictions - test_labels)
# Print out the mean absolute error (mae)
print("Mean Absolute Error:", round(np.mean(errors), 2), "efficiency scores.")

# %%
# Calculate mean absolute percentage error (MAPE)
mape = 100 * (errors / test_labels)
# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print("Accuracy:", round(accuracy, 2), "%.")

# %%
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]
# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot

# Pull out one tree from the forest
tree = rf.estimators_[5]
# Export the image to a dot file
export_graphviz(
    tree, out_file="tree.dot", feature_names=feature_list, rounded=True, precision=1
)
# Use dot file to create a graph
(graph,) = pydot.graph_from_dot_file("tree.dot")
# Write graph to a png file
graph.write_png("tree.png")

# %%
# Limit depth of tree to 3 levels
rf_small = RandomForestRegressor(n_estimators=10, max_depth=3)
rf_small.fit(train_features, train_labels)
# Extract the small tree
tree_small = rf_small.estimators_[5]
# Save the tree as a png image
export_graphviz(
    tree_small,
    out_file="small_tree.dot",
    feature_names=feature_list,
    rounded=True,
    precision=1,
)
(graph,) = pydot.graph_from_dot_file("small_tree.dot")
graph.write_png("small_tree.png")

# %%
# Get numerical feature importances
importances = list(
    rf.feature_importances_
)  # List of tuples with variable and importance
feature_importances = [
    (feature, round(importance, 2))
    for feature, importance in zip(feature_list, importances)
]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
# Print out the feature and importances
[print("Variable: {:20} Importance: {}".format(*pair)) for pair in feature_importances]

# %%
# New random forest with only the two most important variables
rf_most_important = RandomForestRegressor(n_estimators=1000, random_state=42)
# Extract the two most important features
important_indices = [feature_list.index("temp_1"), feature_list.index("average")]
train_important = train_features[:, important_indices]
test_important = test_features[:, important_indices]  # Train the random forest
rf_most_important.fit(
    train_important, train_labels
)  # Make predictions and determine the error
predictions = rf_most_important.predict(test_important)
errors = abs(predictions - test_labels)
# Display the performance metrics
print("Mean Absolute Error:", round(np.mean(errors), 2), "degrees.")
mape = np.mean(100 * (errors / test_labels))
accuracy = 100 - mape
print("Accuracy:", round(accuracy, 2), "%.")

# %%
# Import matplotlib for plotting and use magic command for Jupyter Notebooks
import matplotlib.pyplot as plt

# %matplotlib inline
# Set the style
plt.style.use("fivethirtyeight")
# list of x locations for plotting
x_values = list(range(len(importances)))
# Make a bar chart
plt.bar(x_values, importances, orientation="vertical")
# Tick labels for x axis
plt.xticks(x_values, feature_list, rotation="vertical")
# Axis labels and title
plt.ylabel("Importance")
plt.xlabel("Variable")
plt.title("Variable Importances")

# %%
# Use datetime for creating date objects for plotting
import datetime  # Dates of training values

months = features[:, feature_list.index("month")]
days = features[:, feature_list.index("day")]
years = features[:, feature_list.index("year")]
# List and then convert to datetime object
dates = [
    str(int(year)) + "-" + str(int(month)) + "-" + str(int(day))
    for year, month, day in zip(years, months, days)
]
dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in dates]
# Dataframe with true values and dates
true_data = pd.DataFrame(data={"date": dates, "actual": labels})
# Dates of predictions
months = test_features[:, feature_list.index("month")]
days = test_features[:, feature_list.index("day")]
years = test_features[:, feature_list.index("year")]
# Column of dates
test_dates = [
    str(int(year)) + "-" + str(int(month)) + "-" + str(int(day))
    for year, month, day in zip(years, months, days)
]  # Convert to datetime objects
test_dates = [datetime.datetime.strptime(date, "%Y-%m-%d") for date in test_dates]
# Dataframe with predictions and dates
predictions_data = pd.DataFrame(data={"date": test_dates, "prediction": predictions})
# Plot the actual values
plt.plot(true_data["date"], true_data["actual"], "b-", label="actual")
# Plot the predicted values
plt.plot(
    predictions_data["date"], predictions_data["prediction"], "ro", label="prediction"
)
plt.xticks(rotation="60")
plt.legend()  # Graph labels
plt.xlabel("Date")
plt.ylabel("Maximum Temperature (F)")
plt.title("Actual and Predicted Values")

# %%
# Make the data accessible for plotting
true_data["temp_1"] = features[:, feature_list.index("temp_1")]
true_data["average"] = features[:, feature_list.index("average")]
true_data["friend"] = features[:, feature_list.index("friend")]
# Plot all the data as lines
plt.plot(true_data["date"], true_data["actual"], "b-", label="actual", alpha=1.0)
plt.plot(true_data["date"], true_data["temp_1"], "y-", label="temp_1", alpha=1.0)
plt.plot(true_data["date"], true_data["average"], "k-", label="average", alpha=0.8)
plt.plot(
    true_data["date"], true_data["friend"], "r-", label="friend", alpha=0.3
)  # Formatting plot
plt.legend()
plt.xticks(rotation="60")
# Lables and title
plt.xlabel("Date")
plt.ylabel("Maximum Temperature (F)")
plt.title("Actual Max Temp and Variables")


# %%
