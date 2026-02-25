import pandas as pd
from sklearn.utils import resample
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("Iris.csv")

print(df.head())
print(df["Species"].value_counts().sort_index())

# encode classes
le = LabelEncoder()
df["class"] = le.fit_transform(df["Species"].to_numpy())
print(df.head())


# bootstraping
n_iterations = 200
n_size = df["class"].count()
features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm", "class"]

for i in range(n_iterations):
    train = resample(df[features].to_numpy(), replace=True, n_samples=n_size)
    oob_indices = np.array([i for i in range(len(data)) if i not in train])
    oob = data[oob_indices]

    x_train = train[:, :-1]
    y_train = train[:, -1]
    x_test = oob[:, :-1]
    y_test = obb[:, -1]

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)




import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load a sample dataset (replace with your data loading)
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
data = np.hstack((X, y.reshape(-1, 1))) 

n_iterations = 100 
stats = list() 
n_size = len(data) 

for i in range(n_iterations):
    train = resample(data, replace=True, n_samples=n_size)
    oob_indices = np.array([i for i in range(len(data)) if i not in train]) # A more efficient OOB identification method might be needed for large datasets.
    oob = data[oob_indices]

    if len(oob) > 0:
        X_train = train[:, :-1]
        y_train = train[:, -1]
        X_test = oob[:, :-1]
        y_test = oob[:, -1]

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        stats.append(score)

stats = np.array(stats)

mean_score = np.mean(stats)
std_score = np.std(stats)
print(f'Mean OOB Accuracy: {mean_score:.3f}')
print(f'Standard Deviation: {std_score:.3f}')

alpha = 0.95
p = ((1.0 - alpha) / 2.0) * 100
lower = np.percentile(stats, p)
p = (alpha + ((1.0 - alpha) / 2.0)) * 100
upper = np.percentile(stats, p)
print(f'{alpha*100:.1f}% Confidence Interval: [{lower:.3f}, {upper:.3f}]')

plt.hist(stats, bins=20)
plt.title('Distribution of Bootstrap OOB Accuracy Scores')
plt.xlabel('Accuracy Score')
plt.ylabel('Frequency')
plt.show()
