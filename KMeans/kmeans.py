from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns ; sns.set()
import numpy 
import pandas

iris = datasets.load_iris()
print ("_________________________________________________________________________")
print ("Iris Description" , iris['DESCR'])
print ("_________________________________________________________________________")

#Printing the dataset
print (iris.feature_names)
print ("Total number of instances in iris dataset : " , numpy.size(iris.data))
print ("_________________________________________________________________________")

iris_data = pandas.DataFrame(iris.data)
iris_data.columns = ['Sepal_Length' , 'Sepal_Width' , 'Petal_Length' , 'Petal_Width' ]
iris_target = pandas.DataFrame(iris.target)
iris_target.columns = ['Target']

#Splitting the dataset
features_train, features_test , labels_train , labels_test = train_test_split(iris_data , iris_target , train_size = 0.7)

print("Total number of instances in training dataset : " ,numpy.size(features_train))
print("Total number of instances in test dataset : " ,numpy.size(features_test))


#Model generation
model = KMeans(n_clusters = 3)
model.fit(features_train)
labels_pred = model.predict(features_test)

print("Total number of labels in training dataset: " ,numpy.size(labels_train))
print("Total number of labels predicted: " ,numpy.size(labels_pred))
print("Total number of labels originally: " ,numpy.size(labels_test))

#Viewing the results
plt.figure(figsize = (20,10))
#############Sepal Information
#Original Graph
plt.subplot(2,2,1)
plt.scatter(features_test.Sepal_Length , features_test.Sepal_Width , c=labels_test.Target , cmap=plt.cm.copper)
plt.title("Original Classification of Sepal Information")

plt.subplot(2,2,2)
plt.scatter(features_test.Sepal_Length , features_test.Sepal_Width , c=labels_pred , cmap = plt.cm.copper)
plt.title("K Means Classification of Sepal Information")
plt.show()

#Viewing the results
plt.figure(figsize = (20,10))
#############Petal Information
#Original Graph
plt.subplot(2,2,1)
plt.scatter(features_test.Petal_Length , features_test.Petal_Width , c=labels_test.Target , cmap=plt.cm.copper)
plt.title("Original Classification of Petal Information")

plt.subplot(2,2,2)
plt.scatter(features_test.Petal_Length , features_test.Petal_Width , c=labels_pred , cmap = plt.cm.copper)
plt.title("K Means Classification of Petal Information")
plt.show()

mat = confusion_matrix(labels_test.Target , labels_pred  )
sns.heatmap(mat.T ,square=True, annot=True, fmt='d', cbar=False,
            xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('True Labels')
plt.ylabel('Predicted Labels')
plt.title("Confusion matrix")
plt.show()
##To show all color maps in plot
##plt.cm.datad.keys()
