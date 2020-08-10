from csv import reader
from random import seed
from random import randrange
from sklearn import preprocessing 
from math import sqrt
from math import exp
from math import pi
import pandas as pd
import math
import numpy as np

#counting how many correct
def Count_corIncor_classy(correct,incorrect):
    x = {correct,incorrect}
    #print('\nCorrectly Classified Instances:',correct,'\nIncorrectly Classified Instances : ',incorrect)
    
    

# Reading the data set 
def read_File(name):
    data_S = list()
    with open(name, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            data_S.append(row)
    return data_S
	
def K_fold_crossValidate(data_S, algorithm, n_folds, *args):
	folds = crossValidate(data_S, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = accuracy_metric(actual, predicted)
		scores.append(accuracy)
	return scores




# Function to change string column to float
def convert_float(data_S, column):
	for row in data_S:
		row[column] = float(row[column])

# Convert string column to integer
def Col2int(data_S, column):
	class_values = [row[column] for row in data_S]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in data_S:
		row[column] = lookup[row[column]]
	return lookup

# Split a data_S into k folds
def crossValidate(data_S, n_folds):
	data_S_split = list()
	data_S_copy = list(data_S)
	fold_size = int(len(data_S) / n_folds)
	for _ in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(data_S_copy))
			fold.append(data_S_copy.pop(index))
		data_S_split.append(fold)
	return data_S_split
# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
    correct = 0
    incorrect=0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
        else:
            incorrect +=1
    Count_corIncor_classy(correct,incorrect)
    return correct / float(len(actual)) * 100.0

# Evaluate an algorithm using a cross validation split



# Calculate the mean of a list of numbers
def mean(numbers):
	return sum(numbers)/float(len(numbers))

# Calculate the standard deviation of a list of numbers
def stdev(numbers):
	avg = mean(numbers)
	variance = sum([(x-avg)**2 for x in numbers]) / float(len(numbers)-1)
	return sqrt(variance)
# Calculate the mean, stdev and count for each column in a data_S
def summarize_data_S(data_S):
	summaries = [(mean(column), stdev(column), len(column)) for column in zip(*data_S)]
	del(summaries[-1])
	return summaries



# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    #exponent = exp(-((x-mean)**2 / (2 * stdev**2 )))
    
    try:
        exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2)))) 
    except ZeroDivisionError:
        exponent =0.00001 #or whatever
    try:
        ans = (1 / (sqrt(2 * pi) * stdev)) * exponent
    except ZeroDivisionError:
        ans =0 #or whatever
    return ans

# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
	total_rows = sum([summaries[label][0][2] for label in summaries])
	probabilities = dict()
	for class_value, class_summaries in summaries.items():
        
        # Applied log
		probabilities[class_value] = summaries[class_value][0][2]/float(total_rows)
		for i in range(len(class_summaries)):
			mean, stdev, _ = class_summaries[i]
            #Applied log
            
			probabilities[class_value] += np.log(calculate_probability(row[i], mean, stdev))
	return probabilities

# Predict the class for a given row
def predict(summaries, row):
	probabilities = calculate_class_probabilities(summaries, row)
	best_label, best_prob = None, -1
	for class_value, probability in probabilities.items():
		if best_label is None or probability > best_prob:
			best_prob = probability
			best_label = class_value
	return best_label
# Naive Bayes Algorithm
def naive_bayes(train, test):
	summarize = summarize_by_class(train)
	predictions = list()
	for row in test:
		output = predict(summarize, row)
		predictions.append(output)
	return(predictions)

# Split data_S by class then calculate statistics for each row
def summarize_by_class(data_S):
	separated = class_Apart(data_S)
	summaries = dict()
	for class_value, rows in separated.items():
		summaries[class_value] = summarize_data_S(rows)
	return summaries

# Split the data_S by class values, returns a dictionary
def class_Apart(data_S):
	separated = dict()
	for i in range(len(data_S)):
		vector = data_S[i]
		class_value = vector[-1]
		if (class_value not in separated):
			separated[class_value] = list()
		separated[class_value].append(vector)
	return separated
    

Dataset_Name='hayes-roth.data'
HRdata=read_File(Dataset_Name)
#HRdata[0][0]=92
#print(HRdata)
seed(75)
for i in range(len(HRdata[0])-1):
    convert_float(HRdata, i)
# convert class column to integers
Col2int(HRdata, len(HRdata[0])-1)
# fit model
model = summarize_by_class(HRdata)
# define a new record
row = [1,1,1,2,1]
# predict the label
label = predict(model, row)
#print('Data=%s, Predicted: %s' % (row, label))
# evaluate algorithm
n_folds = 10
scores = K_fold_crossValidate(HRdata, naive_bayes, n_folds)
print('-----------------------Hayes-Roth Data Set--------------------------\n')
print('Test Data = %s      Predicted Class =  %s' % (row, label))
print('\nAccuracy using 10-Fold cross validation : \n%s' % scores)
print('\nMean Accuracy of  Naive Bayes algorithm =  %.3f%%' % (sum(scores)/float(len(scores))))


#CAR.........



Cdata= pd.read_csv('car.csv',header=None)

Cdata[0] = Cdata[0].replace({"vhigh": 4, "high": 3, "med":2, "low": 1})
Cdata[1] = Cdata[1].replace({"vhigh": 4, "high": 3, "med":2, "low": 1})
Cdata[2] = Cdata[2].replace({"5more": 5})
Cdata[3] = Cdata[3].replace({"more": 6})
Cdata[4] = Cdata[4].replace({"small": 1, "med":2, "big": 3})
Cdata[5] = Cdata[5].replace({"high": 3, "med":2, "low": 1})
Cdata[6] = Cdata[6].replace({"vgood": 4, "good": 3, "acc":2, "unacc": 1})
Cdata.head(5)

Cdata = Cdata.to_numpy().tolist()
seed(2)
for i in range(len(Cdata[0])-1):
    convert_float(Cdata, i)
# convert class column to integers
Col2int(Cdata, len(Cdata[0])-1)
# fit model
model = summarize_by_class(Cdata)
# define a new record
row = [4,4,2,2,1,2]
# predict the label
label = predict(model, row)
#print('Data=%s, Predicted: %s' % (row, label))
# evaluate algorithm
n_folds = 10
scores = K_fold_crossValidate(Cdata, naive_bayes, n_folds)
print('\n-----------------------Car Evaluation Data Set--------------------------\n')
print('Test Data = %s      Predicted Class =  %s' % (row, label))
print('\nAccuracy using 10-Fold cross validation : \n%s' % scores)
print('\nMean Accuracy of  Naive Bayes algorithm =  %.3f%%' % (sum(scores)/float(len(scores))))



# Cancer data set with data set label ecoding
BCdata= pd.read_csv('breast-cancer.csv',header=None)
BCdata = BCdata[[1,2,3,4,5,6,7,8,9,0]]
BCdata.head(5)
# Cancer data set with data set label ecoding
# Import label encoder 

  
# label_encoder object knows how to understand word labels. 
label_encoding = preprocessing.LabelEncoder() 
  
# Encode labels in column 0. 
BCdata[0]= label_encoding.fit_transform(BCdata[0]) 
  
BCdata[0].unique()

# Encode labels in column 1. 
BCdata[1]= label_encoding.fit_transform(BCdata[1]) 
# Encode labels in column 2. 
BCdata[2]= label_encoding.fit_transform(BCdata[2]) 

# Encode labels in column 3. 
BCdata[3]= label_encoding.fit_transform(BCdata[3]) 

# Encode labels in column 4. 
BCdata[4]= label_encoding.fit_transform(BCdata[4]) 
# Encode labels in column 5. 
BCdata[5]= label_encoding.fit_transform(BCdata[5]) 
# Encode labels in column 6. 
BCdata[6]= label_encoding.fit_transform(BCdata[6]) 
# Encode labels in column 7. 
BCdata[7]= label_encoding.fit_transform(BCdata[7]) 
# Encode labels in column 8. 
BCdata[8]= label_encoding.fit_transform(BCdata[8]) 
# Encode labels in column 9. 
BCdata[9]= label_encoding.fit_transform(BCdata[9]) 
  
BCdata.head(15)
BCdata = BCdata.to_numpy().tolist()

seed(5)
#for i in range(len(BCdata[0])-1):
    #convert_float(Cdata, i)
# convert class column to integers
#Col2int(BCdata, len(BCdata[0])-1)
# fit model
model = summarize_by_class(BCdata)
# define a new record
row = [2,2,5,0,1,2,0,3,0]
# predict the label
label = predict(model, row)
print('\n-----------------------Breast Cancer Data Set--------------------------\n')
print('Test Data = %s      Predicted Class =  %s' % (row, label))
# evaluate algorithm
n_folds = 10
scores = K_fold_crossValidate(BCdata, naive_bayes, n_folds)
print('\nAccuracy using 10-Fold cross validation : \n%s' % scores)
print('\nMean Accuracy of  Naive Bayes algorithm =  %.3f%%' % (sum(scores)/float(len(scores))))