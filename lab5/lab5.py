from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import graphviz

df = pd.read_csv("data.csv", sep=" ")
print(df.head())
print(df.shape)

# Label Encode of all features
weather_le = LabelEncoder()
df["Weather"] = weather_le.fit_transform(df["Weather"])

parents_le = LabelEncoder()
df["Parents"] = parents_le.fit_transform(df["Parents"])

cash_le = LabelEncoder()
df["Cash"] = cash_le.fit_transform(df["Cash"])

exam_le = LabelEncoder()
df["Exam"] = exam_le.fit_transform(df["Exam"])

decision_le = LabelEncoder()
df["Decision"] = decision_le.fit_transform(df["Decision"])

# split
X = df[["Weather", "Parents", "Cash", "Exam"]]
y = df["Decision"]

clf = DecisionTreeClassifier()
clf = clf.fit(X, y)

plot_tree(clf)
plt.title("Decision tree trained on all the iris features")
plt.savefig('tree.png', dpi=300, bbox_inches='tight')

# extra vis
dot_data = export_graphviz(clf, out_file=None,
                     feature_names=["Weather", "Parents", "Cash", "Exam"],
                     class_names=decision_le.classes_,
                     filled=True, rounded=True,
                     special_characters=True)
graph = graphviz.Source(dot_data)