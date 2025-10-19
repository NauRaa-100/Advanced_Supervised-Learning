
#  Machine Learning Algorithms: KNN, Naive Bayes, and SVM

This repository contains practical implementations of **three core machine learning algorithms** for classification and regression tasks  **K-Nearest Neighbors (KNN)**, **Naive Bayes**, and **Support Vector Machines (SVM)**.  
Each algorithm is implemented from scratch using **scikit-learn**, with clear examples, synthetic datasets, and visualizations.

---

##  Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [ K-Nearest Neighbors (KNN)](#-k-nearest-neighbors-knn)
- [ Naive Bayes](#-naive-bayes)
- [ Support Vector Machine (SVM)](#-support-vector-machine-svm)
- [ Datasets Used](#-datasets-used)
- [ Results & Visualizations](#-results--visualizations)
- [ Future Work](#-future-work)

---

##  Overview
The goal of this project is to:
- Understand **how classical ML algorithms work**.
- Learn **how to visualize decision boundaries**.
- Compare **model behavior on different data distributions**.
- Build confidence in using `scikit-learn` for practical experimentation.
---

**Requirements**

```
numpy  
matplotlib  
scikit-learn  
```

---


##  K-Nearest Neighbors (KNN)

KNN is a **non-parametric, instance-based** learning algorithm.
It predicts the label of a data point by looking at the **K closest training samples** in the feature space.

**Example used:**

* Synthetic blob datasets (2 or 3 classes)
* Classification visualization with decision boundaries

```python
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

X, y = make_blobs(n_samples=200, centers=3, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("Accuracy =>", accuracy_score(y_test, knn.predict(X_test)))
```

*The number of neighbors (K) directly affects bias–variance tradeoff.*

---

##  Naive Bayes

Naive Bayes is a **probabilistic classifier** based on **Bayes’ theorem**, assuming independence between features.

**Example used:**

* Synthetic dataset with 2 features and categorical target
* GaussianNB (continuous features)
* Comparison of predicted vs true classes

```python
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=300, n_features=2, n_informative=2, n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
print("Accuracy =>", accuracy_score(y_test, model.predict(X_test)))
```

 *Naive Bayes performs surprisingly well even when independence assumptions are violated.*

---

##  Support Vector Machine (SVM)

SVM tries to find the **optimal hyperplane** that best separates classes in the feature space.

### Kernels Used:

* **Linear Kernel:** for linearly separable data
* **RBF Kernel:** for non-linear patterns (moons, circles)
* **Polynomial Kernel:** for multi-class data

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)
print("Accuracy =>", accuracy_score(y_test, svm.predict(X_test)))

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(X_scaled[:,0].min()-1, X_scaled[:,0].max()+1, 300),
                     np.linspace(X_scaled[:,1].min()-1, X_scaled[:,1].max()+1, 300))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=y, cmap='coolwarm', edgecolor='k')
plt.title("SVM Decision Boundary (RBF Kernel)")
plt.show()
```

*SVMs are powerful and work well in both low- and high-dimensional spaces.*

---

##  Datasets Used

* `make_blobs`: Simple cluster-based datasets
* `make_moons`: Non-linear separable data
* `make_circles`: Concentric circular data
* `make_classification`: Random classification dataset
* `numpy` synthetic examples (for XOR pattern)

---

##  Results & Visualizations

| Algorithm   | Kernel/Type | Dataset             | Accuracy (approx.) | Notes                               |
| ----------- | ----------- | ------------------- | ------------------ | ----------------------------------- |
| KNN         | —           | make_blobs          | 90–98%             | Simple classification               |
| Naive Bayes | Gaussian    | make_classification | 85–95%             | Fast, works well on simple features |
| SVM         | Linear      | make_blobs          | 95–100%            | Best on linearly separable data     |
| SVM         | RBF         | make_moons          | 90–97%             | Captures complex curves             |
| SVM         | Poly        | 3-class blobs       | 85–95%             | Handles multi-class problems        |

---

##  Future Work

* Implement **GridSearchCV** for hyperparameter tuning (C, gamma, degree).
* Add **visual comparison** between algorithms.
* Include **Support Vector Regression (SVR)** examples.
* Extend to **real-world datasets (Iris, Titanic, MNIST)**.


---

##  Author

**Nau Raa**
  ML & AI Learner | Passionate about data science and automation
  Feel free to connect or open issues for collaboration!


```
