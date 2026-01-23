# import required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pprint import pprint

# load data
# data was downloaded from: https://www.kaggle.com/datasets/jayaantanaath/student-habits-vs-academic-performance?resource=download
df = pd.read_csv("student_habits_performance.csv")


print(f"Number of observation: {df.shape[0]}")
features = df.columns
print(f"Number of features: {df.shape[1]} which are {[i for i in features.to_list()]}")
target = "exam_score"
print(f"Target feature: {target}")
numerical_df = df.select_dtypes(include=['number'])
print(f"Numerical Features: {numerical_df.columns.to_list()}")
object_df = df.select_dtypes(include=['object'])
print(f"Categorical Features:{object_df.columns.to_list()}")

# statistical summary of the dataset
print("Details about all Variables:")
print(df.describe(include="all"))

######################
# Preprocessing
######################

# convert the exam score to grade
def convert_to_grade(score):
    if score >= 91:
        return "O"
    elif score >= 85:
        return "A+"
    elif score >= 75:
        return "A"
    elif score >= 65:
        return "B+"
    elif score >= 60:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 50:
        return "P"
    else:
        return "F"

df["grade"] = df[target].apply(convert_to_grade)

# remove any missing values by elimination
print(df.isna().sum())
sns.heatmap(df.isnull(), cbar=False)
plt.show()

# remove outliers

# Split and prepare for training
y = df["grade"]
excluded_columns = ["grade", "exam_score"]
columns_to_keep = df.columns[~df.columns.isin(excluded_columns)]

# Select only the columns to keep
x = df[columns_to_keep]