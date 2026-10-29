#Importing Libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, accuracy_score

#Importing dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
#Using pandas to read the dataset from the URL and assign column names
iris_data = pd.read_csv(url, names=column_names)

sns.pairplot(iris_data, hue="class") #Pairplot to visualize the relationships between features
plt.show() #Show the plot

x = iris_data.drop("class", axis=1) #Features
y = iris_data['class'] #Target variable

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42) #Splitting the dataset into training and testing sets

print(x)
print(y)

#print(iris_data.describe()) #Description of the dataset 

knn = KNeighborsClassifier(n_neighbors= 3) #Creating KNN model with 3 neighbors
knn.fit(X_train, y_train) #Fitting the model to the training data
y_pred = knn.predict(X_test) #Making predictions on the test data
print("Accuracy:", accuracy_score(y_test, y_pred)) #Printing the accuracy of the model


print(classification_report(y_test, y_pred)) #Printing the classification report

X_test.head(2)
new_data = pd.DataFrame({"sepal_length" : [5.7], "sepal_width" : [4.5], "petal_length" : [3.4], "petal_width" : [0.9]}) #Creating a new DataFrame with new data for prediction
print(knn.predict(new_data)) #Making prediction on the new data