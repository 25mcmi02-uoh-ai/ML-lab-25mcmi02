import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import numpy as np

data = {
    "Feature1": [2.5, 3.5, 1.5, 6.5, 7.5, 5.5, 3.0, 6.0, 4.0, 5.0],
    "Feature2": [3.0, 4.5, 2.0, 7.0, 8.0, 5.0, 3.5, 6.5, 4.5, 5.5],
    "Actual Class": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

print(df.head())

# plot features
sns.scatterplot(data=df, x="Feature1", y="Feature2", hue="Actual Class")
plt.show()

# train a logestic regrission model
cls = LogisticRegression(solver="liblinear", max_iter=1)
cls.fit(df[["Feature1", "Feature2"]], df["Actual Class"])
y_true = df["Actual Class"].to_list()
y_pred = list(cls.predict(df[["Feature1", "Feature2"]]))

print(f"True     : {y_true}\nPredicted: {y_pred}")
print(confusion_matrix(y_true, y_pred))

# selectivly modify the coefficients or intercept for more error


# Function to plot decision boundary
def plot_logitic(b, w1, w2, data):
    c = -b/w2
    m = -w1/w2
    xmin, xmax = 0.0, 10.0
    ymin, ymax = 0.0, 10.0
    xd = np.array([xmin, xmax])
    yd = m*xd + c
    X = np.asarray(data[["Feature1", "Feature2"]])
    Y = np.asarray(data["Actual Class"], dtype="int")
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)
    plt.scatter(*X[Y==0].T, s=8, alpha=0.5)
    plt.scatter(*X[Y==1].T, s=8, alpha=0.5)
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.ylabel(r'$x_2$')
    plt.xlabel(r'$x_1$')
    plt.show()


# Retrieve the model parameters.
b = cls.intercept_[0]
w1, w2 = cls.coef_.T

plot_logitic(b, w1, w2)


# # try Quadratic Decision Boundary
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# model = QuadraticDiscriminantAnalysis()
# model.fit(df[["Feature1", "Feature2"]], df["Actual Class"])
# y_true = df["Actual Class"].to_list()
# y_pred = list(model.predict(df[["Feature1", "Feature2"]]))

# print(f"True     : {y_true}\nPredicted: {y_pred}")
# print(confusion_matrix(y_true, y_pred))

# polynomial Logistic Regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay

# 1. Define the model pipeline
# Degree=2 generates: [1, x1, x2, x1^2, x1x2, x2^2]
model = make_pipeline(
    PolynomialFeatures(degree=2, include_bias=False),
    LogisticRegression()
)
X = df[["Feature1", "Feature2"]]
y = df["Actual Class"]

# 2. Fit to your data (X_train, y_train)
model.fit(X, y)

# plot the curve
# 3. Display the decision boundary

disp = DecisionBoundaryDisplay.from_estimator(
    model, X, 
    response_method="predict", 
    cmap=plt.cm.coolwarm, 
    alpha=0.8
)

# 4. Overlay original data points using Seaborn for easy hue handling
sns.scatterplot(
    x=df["Feature1"], 
    y=df["Feature2"], 
    hue=df["Actual Class"], 
    palette={0: "blue", 1: "red"}, # Matches coolwarm vibes
    edgecolor="black"
)

plt.legend(title="Actual Class")
plt.show()


# plot function
import numpy as np
import matplotlib.pyplot as plt

def plot_custom_curve(coefficients, x_range=(-10, 10), resolution=100):
    """
    Plots a curve based on its polynomial coefficients.
    Args:
        coefficients: List of [cn, ..., c1, c0] (highest power first)
        x_range: Tuple of (start, end)
        resolution: Number of points to calculate
    """
    # Create x values
    x = np.linspace(x_range[0], x_range[1], resolution)
    
    # Calculate y values using the coefficients
    y = np.polyval(coefficients, x)
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'Coefficients: {coefficients}', linewidth=2, color='royalblue')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title("Controllable Polynomial Curve")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

# Example Usage:
# A quadratic curve: y = 1x^2 + 2x + 1
my_coeffs = [1, 2, 1] 
plot_custom_curve(my_coeffs)
