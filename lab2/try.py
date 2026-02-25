import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("student_habits_performance.csv")
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

df["grade"] = df["exam_score"].apply(convert_to_grade)
print(df["grade"].head())
df['grade'] = df['grade'].astype('category')

print(df.dtypes)
df.dropna()
sns.pairplot(data=df[["age", "gender", "study_hours_per_day", "attendance_percentage", "grade"]], hue="grade")
plt.show()
