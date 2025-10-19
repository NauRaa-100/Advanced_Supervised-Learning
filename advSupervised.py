"""
-- Advance Supervised ..
-- K-Nearest Neighbors (KNN)	
-- Support Vector Machine (SVM)
-- Naive Bayes	

"""

from sklearn.datasets import load_iris
#from sklearn.datasets import load_boston
from sklearn import datasets 
import pandas as pd 
import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB ,GaussianNB,BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVR


#-- Classification ..

#  Load dataset
data = load_iris()
X = data.data
y = data.target

#  Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#  Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#  Train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

#  Predict
y_pred = knn.predict(X_test_scaled)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print('-------------Seperate----------------') 

#-- Regression

'''
# Load dataset
X, y = load_boston(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

print("MSE:", mean_squared_error(y_test, y_pred))

'''
print('-------------Seperate----------------') 


X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=1.5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(kernel='linear', C=1.0, random_state=42)
svm.fit(X_train_scaled, y_train)

y_pred = svm.predict(X_test_scaled)


print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap='coolwarm', alpha=0.7)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# Create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 50)
yy = np.linspace(ylim[0], ylim[1], 50)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = svm.decision_function(xy).reshape(XX.shape)

# Plot decision boundary
ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.title("SVM Decision Boundary")
plt.show()


print('-------------Seperate--------------') 


svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_train_scaled, y_train)
y_pred_rbf = svm_rbf.predict(X_test_scaled)

print("RBF Kernel Accuracy:", accuracy_score(y_test, y_pred_rbf))


print('-------------Seperate--------------')  


X = np.linspace(0, 10, 50).reshape(-1, 1)
y = np.sin(X).ravel()

svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
svr.fit(X, y)
y_pred = svr.predict(X)

plt.scatter(X, y, color='gray')
plt.plot(X, y_pred, color='red')
plt.title("Support Vector Regression")
plt.show()

print('-------------Seperate--------------')  

#Model: MultinomialNB



reviews = [
    "I love this movie", "This film was terrible", "Absolutely fantastic story",
    "I hated it", "Best movie ever", "Not good", "I enjoyed every part",
    "Awful acting", "It was okay", "Amazing film", "Worst movie of the year",
    "So boring", "Brilliant!", "Very bad"
]

labels = [1,0,1,0,1,0,1,0,0,1,0,0,1,0]  # 1 = Positive, 0 = Negative

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

x_train, x_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

nb= MultinomialNB()
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)

print('-------------Seperate--------------') 


Model: GaussianNB


# Synthetic dataset
np.random.seed(42)
age = np.random.randint(20, 60, 100)
income = np.random.randint(30000, 120000, 100)
savings = np.random.randint(1000, 30000, 100)
buy_car = np.where((income > 70000) & (savings > 10000), 1, 0)

x = np.column_stack((age, income, savings))
y = buy_car

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42)

nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)
print('Acuuracy => ',accuracy_score(y_test,y_pred))
print('-------------Seperate--------------') 


#Problem 3: Email Spam Detection (binary features)
Model: BernoulliNB


# Each feature = whether a word appears (1) or not (0)
X = np.array([
    [1, 0, 1, 0, 1],  # has "offer", "click", "free"
    [0, 1, 0, 0, 0],
    [1, 1, 1, 0, 1],
    [0, 0, 0, 1, 0],
    [1, 0, 0, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 0, 0, 1, 1]
])

# 1 = Spam, 0 = Not spam
y = np.array([1, 0, 1, 0, 1, 0, 1, 0])

x_train, x_test, y_train, y_test = train_test_split(X
, y, test_size=0.25, random_state=42)

nb=BernoulliNB()
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)

print(" Accuracy => ",accuracy_score(y_test,y_pred))


print('-------------Seperate--------------')  

#Problem 1: Predict Pass/Fail using KNN Classification



np.random.seed(42)
study_hours = np.random.randint(1, 10, 100)
sleep_hours = np.random.randint(4, 9, 100)
pass_exam = ((study_hours > 5) & (sleep_hours >= 6)).astype(int)
df = pd.DataFrame({
    'study_hours': study_hours,
    'sleep_hours': sleep_hours,
    'pass_exam': pass_exam
})

X = df[['study_hours', 'sleep_hours']].values
y = df['pass_exam'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print('Accuracy => ',accuracy_score(y_test,y_pred))
print('-------------Seperate--------------')  

#Problem 2: Predict House Price using KNN Regression

np.random.seed(42)
area = np.random.randint(50, 300, 120)
rooms = np.random.randint(1, 6, 120)
price = 50000 + area * 1200 + rooms * 8000 + np.random.randint(-10000, 10000, 120)

df = pd.DataFrame({'area': area, 'rooms': rooms, 'price': price})

X = df[['area', 'rooms']].values
y = df['price'].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

knn=KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print('Accuracy => ',knn.score(X_test,y_test))
print('-------------Seperate--------------')  

#Problem 4: Binary Classification (synthetic 2D data)


from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=200,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print("Accuracy => ",knn.score(X_test,y_test))

print('-------------Seperate--------------')  

#Problem 5: Visualization dataset for KNN boundary plotting

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, n_features=2, cluster_std=1.0, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)

print("Accuracy => ",knn.score(X_test,y_test))

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='rainbow')
plt.show() 

print('-------------Seperate--------------')  


# Problem 1: Linear SVM Classification

from sklearn.datasets import make_blobs

# Generate data
X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42, cluster_std=1.0)

# Scale & Split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

svm=SVC(kernel='linear',C=1.0,random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("Accuracy => ",accuracy_score(y_test,y_pred))

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='tab10',alpha=0.8,s=50)

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

xx=np.linspace(xlim[0],xlim[1],50)
yy=np.linspace(ylim[0],ylim[1],50)

xx,yy=np.meshgrid(xx,yy)
xy=np.vstack([xx.ravel(),yy.ravel()]).T
z=svm.decision_function(xy).reshape(xx.shape)

ax.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()

print('-------------Seperate--------------')  

# Problem 2: Non-linear SVM (RBF Kernel)


from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

svm = SVC(kernel='rbf', gamma='scale', C=1.0)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)

print("Accuracy => ", accuracy_score(y_test, y_pred))

# ==== Visualization ====
plt.figure(figsize=(7,5))

x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))

Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap='rainbow')
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='rainbow', edgecolors='k')
plt.title("SVM RBF Decision Boundary")
plt.show()

print('-------------Seperate--------------')  

# Problem 3: SVR for Predicting Continuous Values

# Synthetic data: predict salary based on experience
np.random.seed(42)
experience = np.random.randint(1, 20, 100)
education_level = np.random.randint(1, 4, 100)  # 1=Highschool, 2=Bachelor, 3=Master+
salary = 2000 + experience * 400 + education_level * 1000 + np.random.randint(-500, 500, 100)

X = np.column_stack((experience, education_level))
y = salary

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

svr = SVR(kernel='rbf', C=0.1, gamma='scale', epsilon=0.5)
svr.fit(X_train,y_train)
y_pred=svr.predict(X_test)
print("MSE => ",mean_squared_error(y_test,y_pred))
plt.scatter(y_test,y_pred)
plt.show()

print('-------------Seperate--------------')  

# Problem 4: Multi-class SVM

from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42, cluster_std=1.2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

svm = SVC(kernel='poly', degree=3, C=1.0)

svm.fit(X_train,y_train)
y_pred =svm.predict(X_test)
print("Accuracy:", accuracy_score(y_test,y_pred))

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='rainbow',alpha=0.8)
plt.show()


print('-------------Seperate--------------') 


# Problem 1: Linear SVM (2 Classes)
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, centers=2, n_features=2, random_state=42, cluster_std=1.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

#classification 

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='rainbow',alpha=0.8,s=50)

# linear 
svm=SVC(kernel='linear',C=0.1,random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("Accuracy => ",accuracy_score(y_test,y_pred))
ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

xx=np.linspace(xlim[0],xlim[1],50)
yy=np.linspace(ylim[0],ylim[1],50)

xx,yy=np.meshgrid(xx,yy)
xy=np.vstack([xx.ravel(),yy.ravel()]).T
z=svm.decision_function(xy).reshape(xx.shape)
ax.contour(xx, yy, z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
plt.show()

print('-------------Seperate--------------') 



# Problem 2: Non-linear data (moons)
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

#classification 

svm=SVC(kernel='rbf',C=0.1,gamma='scale',random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

x_min,x_max=X_scaled[:,0].min() -1 ,X_scaled[:,0].max() +1
y_min,y_max=X_scaled[:,1].min() -1 ,X_scaled[:,1].max() +1

xx=np.linspace(x_min,x_max,50)
yy=np.linspace(y_min,y_max,50)

xx,yy=np.meshgrid(xx,yy)

xy=np.vstack([xx.ravel(),yy.ravel()]).T

z=svm.predict(xy).reshape(xx.shape)
plt.contourf(xx, yy, z, alpha=0.3, cmap='rainbow')
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='rainbow',alpha=0.8)
plt.show()

print('-------------Seperate--------------') 



# Problem 3: Non-linear data (circles)
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

svm=SVC(kernel='poly',C=0.1,gamma='scale',random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='rainbow')
y_curv=np.linspace(y_test.min(),y_test.max(),90)
plt.plot(y_curv,y_pred)
plt.show()
print('-------------Seperate--------------') 



# Problem 4: Multi-class (3 classes)
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=400, centers=3, n_features=2, random_state=42, cluster_std=1.2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
#plt.scatter(X_scaled[:,0],X_scaled[:,1])
#plt.show()

svm=SVC(kernel='poly',C=0.1,gamma='scale',random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print('Accuracy => ',accuracy_score(y_test,y_pred))
plt.plot(y_test,y_pred,c='pink')
plt.show()
print('-------------Seperate--------------')  

# Problem 5: Support Vector Regression (SVR)

np.random.seed(42)

experience = np.random.randint(1, 20, 100)
education_level = np.random.randint(1, 4, 100)  # 1=Highschool, 2=Bachelor, 3=Master+
salary = 2000 + experience * 400 + education_level * 1000 + np.random.randint(-500, 500, 100)

X = np.column_stack((experience, education_level))
y = salary

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=42)

#plt.scatter(X_scaled[:,0],X_scaled[:,1])
#plt.show()

svm=SVR(kernel='rbf',C=10,gamma='scale')
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("Accuracy => ",svm.score(X_test,y_test))

print('-------------Seperate--------------')  


# ====================================================
# 🧩 Problem 1: Linear Classification (Linearly Separable)
# ====================================================
from sklearn.datasets import make_blobs

# Generate simple 2-class linear dataset
X, y = make_blobs(n_samples=200, centers=2, n_features=2,
                  random_state=42, cluster_std=1.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42)

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='coolwarm',alpha=0.8,s=50)

svm=SVC(kernel='linear',C=0.1,gamma='scale',random_state=42)
svm.fit(X_train,y_train)

y_pred=svm.predict(X_test)
print("Accuracy => ",accuracy_score(y_test,y_pred))

ax=plt.gca()
xlim=ax.get_xlim()
ylim=ax.get_ylim()

xx=np.linspace(xlim[0],xlim[1],200)
yy=np.linspace(ylim[0],ylim[1],200)

xx,yy=np.meshgrid(xx,yy)

xy=np.vstack([xx.ravel(),yy.ravel()]).T
z=svm.decision_function(xy).reshape(xx.shape)

ax.contour(xx,yy,z,c='k',levels=[-1,0,1],linestyles=['--','-','--'])

plt.show()
sv = svm.support_vectors_
ax.scatter(sv[:,0], sv[:,1], s=100, linewidth=1, facecolors='none', edgecolors='k', label='support vectors')
plt.show()


print('-------------Seperate--------------')  


# ====================================================
#  Problem 2: Non-linear Data (Moons)
# ====================================================
from sklearn.datasets import make_moons

# Generate moon-shaped data
X, y = make_moons(n_samples=300, noise=0.25, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

#plt.scatter(X_scaled[:,0],X_scaled[:,1])
#plt.show()
svm=SVC(kernel='rbf',C=10,gamma='scale',random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print('Accuracy => ',accuracy_score(y_test,y_pred))

x_min,x_max=X_scaled[:,0].min() -1 ,X_scaled[:,0].max() +1
y_min,y_max=X_scaled[:,1].min() -1 , X_scaled[:,1].max() +1

xx=np.linspace(x_min,x_max,50)
yy=np.linspace(y_min,y_max,50)

xx,yy=np.meshgrid(xx,yy)

xy=np.vstack([xx.ravel(),yy.ravel()]).T
z=svm.predict(xy).reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.5,cmap='rainbow')
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='coolwarm',alpha=0.8,s=50)
plt.show()
print('-------------Seperate--------------')  


# ====================================================
#  Problem 3: Circular Data (Concentric Circles)
# ====================================================
from sklearn.datasets import make_circles

# Generate circle-shaped data
X, y = make_circles(n_samples=300, noise=0.1, factor=0.5, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

svm=SVC(kernel='rbf',gamma='scale',C=0.1,random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("Accuracy => ",accuracy_score(y_test,y_pred))

x_min,x_max = X_scaled[:,0].min() -1 , X_scaled[:,0].max() +1 
y_min,y_max = X_scaled[:,1].min() -1 , X_scaled[:,1].max() +1 

xx=np.linspace(x_min,x_max,50)
yy=np.linspace(y_min,y_max,50)

xx,yy=np.meshgrid(xx,yy)

xy=np.vstack([xx.ravel(),yy.ravel()]).T

z=svm.predict(xy).reshape(xx.shape)

plt.contourf(xx,yy,z,alpha=0.5,cmap='rainbow')
plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='coolwarm',alpha=0.8,s=50)
plt.show()
print('-------------Seperate--------------')  


# ====================================================
# Problem 4: Multi-class Classification (3 Classes)
# ====================================================
from sklearn.datasets import make_blobs

# Generate 3-class data
X, y = make_blobs(n_samples=400, centers=3, n_features=2,
                  random_state=42, cluster_std=1.2)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

#plt.scatter(X_scaled[:,0],X_scaled[:,1])
#plt.show()
svm=SVC(kernel='poly',gamma='scale',C=0.1,random_state=42)
svm.fit(X_train,y_train)
y_pred=svm.predict(X_test)

print("Accuracy => ",accuracy_score(y_test,y_pred))

plt.scatter(X_scaled[:,0],X_scaled[:,1],c=y,cmap='rainbow')
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm)
disp.plot(cmap='Blues')
plt.show()
plt.show()
print('-------------Seperate--------------')  

# ====================================================
#  Problem 6: XOR Dataset (Difficult Non-linear Separation)
# ====================================================

# XOR Pattern (0,0)=0 , (1,1)=0 , (0,1)=1 , (1,0)=1
X = np.array([[0,0],[0,1],[1,0],[1,1]] * 50)
y = np.array([0,1,1,0] * 50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42)

print('-------------Seperate--------------')  

