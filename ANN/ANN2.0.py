import pandas
import numpy
import imp
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

dataset = datasets.load_diabetes()
print(dataset.data[:6,:])

feature = dataset.data ;
label = dataset.target ;

feature_train , feature_test , label_train , label_test = train_test_split(feature , label , train_size = 0.7)

model = MLPClassifier(hidden_layer_sizes =(5,2) , shuffle = True , max_iter =100)
model.fit(feature_train , label_train)
label_pred = model.predict(feature_test)
print ("Accuracy Score = " , accuracy_score(label_test , label_pred)*100)
