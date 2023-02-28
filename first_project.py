import pandas as pd
from ydata_profiling import ProfileReport
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np

# If there is any new modules that you have never installed in your computer
# please pip install <name of the module> before importing it.

df = pd.read_csv("data.csv")

# profile = ProfileReport(df, title = "Project Report")
# profile.to_file("data_report.html")

#! 1. Preprocessing

# Remove unnecessary columns

df.drop(["customer_id", "tenure"], inplace=True, axis=1)

# Split the data into feature and label variables

x = df.drop("churn", axis=1)

# This is label column
y = df["churn"]

# Convert some columns from numeric type to categorical type
# (active member and credit card should be categorical type instead of numeric)

x["active_member"] = x["active_member"].astype("category")

x["credit_card"] = x["credit_card"].astype("category")

#
#  todo Split the data
# Use the random state argument to make sure we have the same train, test split every time
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

# Define the numeric and categorical features

num_features = ["credit_score", "age", "balance", "products_number", "estimated_salary"]

cat_features = ["country", "gender", "credit_card", "active_member"]

# todo Create a preprocessor to handle categorical and numeric features.
# i will explain more about the advantage of creating a pipeline to train a machine learnign later.

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(), cat_features),
    ]
)

#! 2. Define the model architecture
# for the last layer, use 1 layer since we are dealing with binary classification
# Result of the last layer is predicted probability.
# If it is close to one, it means that it's likely that the customer will churn

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Dense(32, activation="relu", input_shape=(14,)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)

#! 3. Compile the model
# In this compile step, we use Adam algorithm to optimize node parameters.
# for the loss function, binary_crossentropy is used to calculate loss.
# For accuracy, the metrics accuracy is used to evaluate the model's performance

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# todo Create a pipeline
# Create a pipeline to connect the preprocessor with the model structure.
# Teh pipeline starts with the preprocessor and ends with the model algorithm.

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])

#! 4. Train the model
# After defining the structure and compiling it, we should be good to proceed ahead and train the model
# in this training process, we use ten epochs and a batch size of 128
# 10 epochs mean we would interate the entire train data set 10 times
# a batch size of 128 means for every 128 records, the model parameters will be updated.

pipeline.fit(x_train, y_train, model__epochs=10, model__batch_size=128)

# todo Making prediction

y_pred = pipeline.predict(x_test)

#! 5. Evaluation

# ? Evaluation 1 (may not be required for the presentation)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib
matplotlib.use("TkAgg")

# moved down

# ? Evaluation 2
# ! Result from this evaluation is important.

# todo Lift Ratio
import kds

kds.metrics.plot_cumulative_gain(y_test, y_pred)
plt.savefig("gains_chart.png")



# Obtain the result for false positive rate and true positive rate
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

# todo AUC and ROC curve
roc_auc = roc_auc_score(y_test, y_pred)

# Plot the ROC curve

plt.plot(fpr, tpr, label="AUC = {:.2f}".format(roc_auc))
plt.legend()
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.savefig("roc.png")

#! Pick an optimized threshold

# Thinking of a way to optimize the threshold
# there is a trade off between recall and precision. High threshold will lowers false positive but also increases false negative, vice versa.
# TP = TP/(TP+FN)
# FP = FP(FP+TN)
result_data = {"False Positive": fpr, "True Positive": tpr, "Thresholds": thresholds}
result_data = pd.DataFrame(result_data)

# todo plot a 3D graph of tpr, fpr, thresholds

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

axes = plt.axes(projection="3d")
axes.plot3D(
    result_data["False Positive"],
    result_data["True Positive"],
    result_data["Thresholds"],
    "blue",
)
axes.set_title("3D plot for FPR, TPR, and Threshold points")
axes.set_xlabel("False Positive")
axes.set_ylabel("True Positive")
axes.set_zlabel("Threshold")
plt.show()

# According to the plot, we see that the range True Positive between 0.79 and 0.81 would be optimal for our model.

threshold_selected_table = result_data[
    (result_data["True Positive"] > 0.79) & (result_data["True Positive"] < 0.81)
]

# Pick the threshold in the middle 
selected_threshold = threshold_selected_table.iloc[int(len(threshold_selected_table)/2)].Thresholds


# todo Sensitivity and Specificity

# Convert the predicted probabilities to binary class predictions using the selected threshold

y_pred_class = [1 if x >= selected_threshold else 0 for x in y_pred]

sensitivity = recall_score(y_test, y_pred_class).round(3)
# Use values[0] to access the index only
specificity = (
    1 - result_data[result_data["Thresholds"] == selected_threshold]
    ["False Positive"].values[0]).round(3)
print(f"Sensitivity of the model is {sensitivity}")
print(f"Specificity of the model is: {specificity}")


# todo Accuracy
accuracy = accuracy_score(y_test, y_pred_class)

# todo Precision, Recall, and F1-score, Confusion matrix
precision = precision_score(y_test, y_pred_class)
recall = recall_score(y_test, y_pred_class).round(3)
f1 = f1_score(y_test, y_pred_class)
conf_matrix = confusion_matrix(y_test, y_pred_class)
# print the results
print("Accuracy: ", accuracy)
print("Precision: ", precision)
print("Recall: ", recall)
print("F1-score: ", f1)
print(conf_matrix) # The cols represent prediction, rows are actual values. 

# We can also print recall and precision in a shorter way 
report = classification_report(y_test, y_pred_class)
print(report)



