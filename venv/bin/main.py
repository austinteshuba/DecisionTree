# This is exploration of the decision tree algo.

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

path = "https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/drug200.csv"

# The data is about patients and which of 5 different drugs they were described
# Step 1: Read data
df = pd.read_csv(path, delimiter=",")
print(df.head())

print(df.size)

# Step 2: Split data

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df["Drug"]

print(X)

# So when looking at X, we see some categorical variables
# Step 3: We need to switch these to discrete variables for sklearn's decision tree to work
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(["LOW", "NORMAL", "HIGH"])
X[:,2] = le_BP.transform(X[:,2])

le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(["NORMAL", "HIGH"])
X[:,3] = le_Chol.transform(X[:,3])

print(X[:5])

# Step 4: Split up the data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.3, random_state=3)

## There has to be the same number of samples in the different training and testing sets; Also has to be the same dimension
print(X_train.shape[0] == y_train.shape[0])
print(X_test.shape[0] == y_test.shape[0])

# Step 5: Modelling
# Inside of the classifier, specify criterion="entropy" so we can see the information gain of each node.
drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
print(drugTree) # it shows the default parameters

drugTree.fit(X_train,y_train)

# Step 6: Prediction
predTree = drugTree.predict(X_test)
print(predTree[0:5])
print(" ")
print(y_test[0:5])

# wow 5/5 pretty good!
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_test, predTree))

# Wow, this is very accurate

# Lets manually do it
# For a decision tree, this is # of correct predictions / # of total predictions
print(y_test.values)
print(predTree)

correctPredictions = 0
for n in range(0,len(y_test.values)):
    if (y_test.values[n] == predTree[n]):
        correctPredictions+=1

print(correctPredictions / len(y_test.values))

# Step 7: Visualize the Tree
# Copied from IBM specialization
# Need proper installs
dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_train), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')










