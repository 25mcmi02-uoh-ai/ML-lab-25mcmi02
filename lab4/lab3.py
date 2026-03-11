import pandas as pd

df = pd.read_csv('marksheet.csv')
print(df.head())

# calculate total marks for each student
df['Total'] = df[['Maths', 'Science', 'English', 'History']].sum(axis=1)

def percentage_label(p):
    if p >= 91: return "C1"
    elif p >= 81: return "C2"
    elif p >= 71: return "C3"
    elif p >= 61: return "C4"
    elif p >= 50: return "C5"
    else: return "F"
    
def percentile_label(p):
    if p >= 91: return "P1"
    elif p >= 81: return "P2"
    elif p >= 71: return "P3"
    elif p >= 61: return "P4"
    elif p >= 50: return "P5"
    else: return "F"
    
def grade_system(p):
    if p >= 91: return "O"
    elif p >= 85: return "A+"
    elif p >= 75: return "A"
    elif p >= 65: return "B+"
    elif p >= 60: return "B"
    elif p >= 55: return "C"
    elif p >= 50: return "P"
    else: return "F"
    
def calculate_percentile(total, n):
    count = len(df[df['Total'] < total])
    percentile = (count / n) * 100
    return percentile


df['percentage'] = (df['Total'] / 400) * 100
df['percentage_label'] = df['percentage'].apply(percentage_label)
n_students = len(df)
df['percentile'] = df['Total'].apply(lambda x: calculate_percentile(x, n_students))
df['percentile_label'] = df['percentile'].apply(percentile_label)
df['grade'] = df['percentage'].apply(grade_system)

# ranking students
df['Rank_percentage'] = df['percentage'].rank(ascending=False).astype(int)
df['Rank_total'] = df['Total'].rank(ascending=False).astype(int)
df['Rank_percentile'] = df['percentile'].rank(ascending=False).astype(int)

print(df[[
    'id', 
    'percentage', 
    'percentile', 
    'grade', 
    'Rank_percentage', 
    'Rank_percentile'
]].sort_values('Rank_percentage').head(10))

grade_df = df[['Maths', 'Science', 'English', 'History', 'grade']].copy()
percentagre_df = df[['Maths', 'Science', 'English', 'History', 'percentage_label']].copy()
percentile_df = df[['Maths', 'Science', 'English', 'History', 'percentile_label']].copy()

# triain for predicting grade based on marks
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

X = grade_df[['Maths', 'Science', 'English', 'History']]
y = grade_df['grade']

le = LabelEncoder()
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report for Grade Prediction:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print("Confusion Matrix for Grade Prediction:")
print(confusion_matrix(y_test, y_pred))

# train for predicting percentage label based on marks
X = percentagre_df[['Maths', 'Science', 'English', 'History']]
y = percentagre_df['percentage_label']
le = LabelEncoder()
y = le.fit_transform(y)

X_train_percentage, X_test_percentage, y_train_percentage, y_test_percentage = train_test_split(X, y, test_size=0.2, random_state=42)
clf1 = DecisionTreeClassifier()
clf1.fit(X_train_percentage, y_train_percentage)
y_pred = clf1.predict(X_test_percentage)
print("Classification Report for Percentage Label Prediction:")
print(classification_report(y_test_percentage, y_pred, target_names=le.classes_))
print("Confusion Matrix for Percentage Label Prediction:")
print(confusion_matrix(y_test_percentage, y_pred))

# train for predicting percentile label based on marks
X = percentile_df[['Maths', 'Science', 'English', 'History']]
y = percentile_df['percentile_label']
le = LabelEncoder()
y = le.fit_transform(y)

X_train_percentile, X_test_percentile, y_train_percentile, y_test_percentile = train_test_split(X, y, test_size=0.2, random_state=42)
clf2 = DecisionTreeClassifier()
clf2.fit(X_train_percentile, y_train_percentile)
y_pred = clf2.predict(X_test_percentile)
print("Classification Report for Percentile Label Prediction:")
print(classification_report(y_test_percentile, y_pred, target_names=le.classes_))
print("Confusion Matrix for Percentile Label Prediction:")
print(confusion_matrix(y_test_percentile, y_pred))

# mix the tests for each model use it to predict using the test of the other models
y_pred_percentage = clf1.predict(X_test_percentile)
print("Classification Report for Percentage Label Prediction on Percentile Test Set:")
print(classification_report(y_test_percentile, y_pred_percentage, target_names=le.classes_))
print("Confusion Matrix for Percentage Label Prediction on Percentile Test Set:")
print(confusion_matrix(y_test_percentile, y_pred_percentage))
y_pred_percentile = clf2.predict(X_test_percentage)
print("Classification Report for Percentile Label Prediction on Percentage Test Set:")
print(classification_report(y_test_percentage, y_pred_percentile, target_names=le.classes_))
print("Confusion Matrix for Percentile Label Prediction on Percentage Test Set:")
print(confusion_matrix(y_test_percentage, y_pred_percentile))