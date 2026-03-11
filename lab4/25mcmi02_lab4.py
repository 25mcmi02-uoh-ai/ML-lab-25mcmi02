import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('marksheet.csv')

print(df.head())

# calculate total marks for each student
df['Total'] = df[['Maths', 'Science', 'English', 'History']].sum(axis=1)

def grade_system(p):
    if p >= 91: return "O"
    elif p >= 85: return "A+"
    elif p >= 75: return "A"
    elif p >= 65: return "B+"
    elif p >= 60: return "B"
    elif p >= 55: return "C"
    elif p >= 50: return "P"
    else: return "F"
    
df['percentage'] = (df['Total'] / 400) * 100
df['grade'] = df['percentage'].apply(grade_system)

# Exam_Type feature addition
df['Exam_Type'] = np.random.choice(['Regular', 'Supplementary'], size=len(df), p=[0.5,0.5])

print(df["Exam_Type"].value_counts())

# change grade if Supplementary to only Pass or Fail
# def convert_grade_for_supplementary(row):
#     if row["Exam_Type"] is "Supplementary":
#         if row["grade"] is "F":
#             return "F"
#         else:
#             return "P"
#     else:
#         return row["grade"]

# df["grade"] = df.apply(convert_grade_for_supplementary, args=(df["Exam_Type"]))

# print([row[1] if row[0] is "Regular" or (row[0] is "Supplementry" and row[1] is "F") else "P" for row in df[["Exam_Type", "grade"]].to_dict()])

# print([item[0] for key,item in df[["grade", "Exam_Type"]].items()])

df["grade"] = np.where(df["Exam_Type"] == "Supplementary", np.where(df["grade"] != "F", "P", "F"), df["grade"]) # finally

###################################
# Training
###################################

X = df[['Gender', 'Age', 'Section', 'Maths', 'Science', 'English', 'History', 'Exam_Type']]
y = df['grade']

le = LabelEncoder()
y = le.fit_transform(y)

# ecoding Gender, Section, Exam_Type
le_gender = LabelEncoder()
X["Gender"] = le_gender.fit_transform(X["Gender"])

le_section = LabelEncoder()
X["Section"] = le_section.fit_transform(X["Section"])

le_type = LabelEncoder()
X["Exam_Type"] = le_type.fit_transform(X["Exam_Type"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(X_test)
print(y_pred)

print("Classification Report for Grade Prediction:")
print(classification_report(y_test, y_pred, target_names=le.classes_))



############################
# User Input and prediction
############################

print("New Record:")
gender = input("Enter Gender (Male or Female): ")
age = input("Enter Age (13 - 15): ")
section = input("Enter Section (A, B, C): ")
section = input("Enter Section (A, B, C): ")
marks = input("Enter marks (Math Science English History): ").split()

pd.DataFrame(
    {
        "Gender": le_gender.transform([gender])[0],
        "Age": age,
        "Section": le_section.transform([section])[0],
        "Math": int(marks[0]),
        "Science": int(marks[1]),
        "English": int(marks[2]),
        "History": int(marks[3]),
        "Exam_Type": "Regular"
    }, index=[0]
)

record = pd.Series([le_gender.transform([gender])[0], age, le_section.transform([section])[0], int(marks[0]), int(marks[1]), int(marks[2]), int(marks[3]), ])
# record = pd.Series([gender, age, section, int(marks[0]), int(marks[1]), int(marks[2]), int(marks[3])])

print(record)
print(X_test.iloc[0,:])
print(clf.predict([X_test.iloc[0,:]]))
print(clf.predict([record]))