from sklearn import datasets    #To get iris dataset
from sklearn import svm         #To fit the svm classifier
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split

#Importing iris dataset to model Svm classifier
iris = datasets.load_iris()
#iris = np.asarray(iris)

#print ("Iris Dataset Description : :" , iris['DESCR'])
print ("------------------------------------------------------")
#print ("Iris Target ::" , iris['target'])

#Visualizing Sepal data
def visulaize_sepal_data() :
    features_train = iris.data[:,:2]
    labels_train = iris['target']
    plt.scatter(features_train[:,0] , features_train[:,1] , c = labels_train , cmap=plt.cm.coolwarm)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    plt.title("Sepal Width & Length")
    plt.show()

#Visualizing Petal Data
def visulaize_petal_data() :
    features_train = iris.data[:,2:]
    labels_train = iris['target']
    plt.scatter(features_train[:,0] , features_train[:,1] , c = labels_train , cmap=plt.cm.coolwarm)
    plt.xlabel("Petal Length")
    plt.ylabel("Patel Width")
    plt.title("Patel Width & Length")
    plt.show()

#Calling defination
visulaize_petal_data()
visulaize_sepal_data()

#Extracting features(Sepal) and labels and buiding model and fitting dataset into model 
features = iris.data[:,:2]
labels = iris['target']

features_train , features_test , labels_train , labels_test = train_test_split(features,labels , train_size = .3)

C = 1.0
model_linear = svm.SVC(kernel = "linear" , C = C)
model_linear.fit(features_train,labels_train)
model_rbf = svm.SVC(kernel = "rbf" ,gamma = 0.7, C = C)
model_rbf.fit(features_train,labels_train)
model_poly = svm.SVC(kernel = "poly" ,degree = 3, C = C)
model_poly.fit(features_train,labels_train)

#create a mesh to plot in
xmin , xmax = features_train[:,0].min()-1 , features_train[:,0].max()+1
ymin , ymax =features_train[:,1].min()-1 , features_train[:,1].max()+1
h = (xmax/xmin)/100    #Step size in the mesh
#A Meshgrid is an ND-coordinate space generated by a set of arrays. Each point on the meshgrid corresponds to a combination of one value from each of the arrays.
#arange - Return evenly spaced values within a given interval.
xx , yy = np.meshgrid(np.arange(xmin , xmax , h) , np.arange(ymin ,ymax, h))


for i , model in enumerate((model_linear , model_rbf , model_poly)) :
    #Subplot- first two parameter specifies the size of the plots and the last parameter decides where that particular subplotwill be placed
    plt.subplot(2,2,i+1)
    #subplot_adjuest : first - the amount of width reserved for space between subplots : second : the amount of height reserved for space between subplots
    plt.subplots_adjust(wspace=0.4,hspace=0.4)

    # title for the plots
    titles = ['SVC with linear kernel',
	    'SVC with RBF kernel',
	    'SVC with polynomial (degree 3) kernel']

    #c_ - concatenating the array results
    #ravel - Flattened array having same type as the Input array and and order as per choice. 
    labels_pred = model.predict(np.c_[xx.ravel() , yy.ravel()])
    #labels_pred = model.predict(features_test)
     
    #reshape - giving the rows and columns value of the 'xx'
    labels_pred = labels_pred.reshape(xx.shape)
 
    #draw contour lines and filled contours,
    plt.contourf(xx,yy,labels_pred ,cmap = plt.cm.tab10 , alpha=0.8)

    plt.scatter(features_train[:,0] , features_train[:,1] , c = labels_train , cmap=plt.cm.coolwarm)
    plt.xlabel("Sepal Length")
    plt.ylabel("Sepal Width")
    #Srtting limits
    plt.xlim(xx.min() , xx.max())
    plt.ylim(yy.min() , yy.max())   
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.show()



















