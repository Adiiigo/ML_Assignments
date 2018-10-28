import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.Random;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

public class naiveBayes {

	public static void main(String[] args) throws Exception {
		// TODO Auto-generated method stub

		//Too accept the value from the file and storing it in buffer
		BufferedReader breader = new BufferedReader(new FileReader(new File("C:/Users/AditiGoyal/Desktop/Mine/ML/datasets/glass.arff"))) ;
	
		//To store the value of the instances(attributes+data) of the training dataset 
		Instances dataset = new Instances(breader) ;
		
		//To set which attribute consists of the class variable
		int numAttr = dataset.numAttributes() ;
		System.out.println("Number of Attributes = " + numAttr);
		System.out.println("Attributes at 0th position is "+ dataset.attribute(0)) ;
		System.out.println("Attributes at 9th position"+ dataset.attribute(9)) ;
		
		//Since Java is 0-index we have to subtract 1
		dataset.setClassIndex(dataset.numAttributes()-1);
		
		//Closing the bufferReader object
		breader.close();
		
		//Randomizing the dataset before splitting so as to obtain better results
		dataset.randomize(new java.util.Random(0));
		
		System.out.println("Total number of instances =" + dataset.numInstances());
		//Splitting that dataset
		int trainSize = (int)Math.round(dataset.numInstances()*70/100) ;
		int testSize = dataset.numInstances() - trainSize ;
		
		System.out.println("TrainSize = " + trainSize + "Test Size = " + testSize);
		
		//second parameter - From . Third parameter = capacity
		Instances train = new Instances(dataset , 0 , trainSize) ;
		Instances test = new Instances(dataset , trainSize, testSize) ;
		
		//Generating naive bayes model
		NaiveBayes model = new NaiveBayes() ;
		
		//Building Classifier on the basis of training dataset
		model.buildClassifier(train) ;
		
		//Generating Evaluating model for testing dataset 
		Evaluation eval = new Evaluation(train) ;
		eval.evaluateModel(model, test) ;		
		System.out.println(eval.toSummaryString());
		//Metrics 
		System.out.println("Correctly Classified Instances: "+eval.correct());
		System.out.println("Error rate: " +eval.errorRate()*100 );
		System.out.println("Pct Correct:" +eval.pctCorrect());
		
		//Predicting the value of the test dataset manually
		for (int i = 0 ; i < test.numInstances() ; i++)
		{
			test.instance(i).setClassMissing(); 
			double cls = model.classifyInstance(test.instance(i)) ;
			test.instance(i).setClassValue(cls);
		}
		
		//Class-wise Metrics display
		for (int i= 0 ; i <dataset.numClasses() ; i++)
		{
			System.out.println("Class " + i);
			System.out.println("Precision: " +eval.precision(i));
			System.out.println("Recall: " +eval.recall(i));
		}
 
		//Applying the naive bayes model without splitting and only applying cross validation
		eval.crossValidateModel(model, dataset, 10, new Random(1));
		System.out.println(eval.toSummaryString("\nResults\n=====\n",true));
		System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
	}

}
 