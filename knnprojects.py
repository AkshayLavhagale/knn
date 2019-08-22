# k-nearest neighbors algorithm (k-NN)
# kneighbors(self[, X, n_neighbors, …])	Finds the K-neighbors of a point.
# kneighbors(self, X=None, n_neighbors=None, return_distance=True)
# Returns:	dist : array - Array representing the lengths to points, only present if return_distance=True
#           ind : array - Indices of the nearest points in the population matrix.

import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing  # sklearn converts non-numeric data to integer using preprocessing.


data = pd.read_csv("car.data")
print(data.head())

# not to use features that have non-numerical data(yes/no). So always convert non-numerical data numerical.

le = preprocessing.LabelEncoder()  # this will convert non-numeric data to integer values.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

x = list(zip(buying, maint, door, persons, lug_boot, safety))  # all features are converts into tuple objects.
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print(acc)

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test, "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("N: ", n)

# here the value of K = 9,which means it measuring magnitude(distance)
# of partiualr point from other nearest possible 9 points.
