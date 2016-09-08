import sklearn
from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree
from sklearn.externals.six import StringIO
import pydot
iris = load_iris()

test_index = [1,50,100]

#training data
train_target = np.delete(iris.target,test_index,0)
train_data = np.delete(iris.data,test_index,0)

#test data
test_target = iris.target[test_index]
test_data = iris.data[test_index]

classifier = tree.DecisionTreeClassifier()
classifier = classifier.fit(train_data,train_target)
print('Decision Tree Output',classifier.predict(test_data))
print ('Original Output From Table',test_target)

dot_data = StringIO()
tree.export_graphviz(classifier,out_file=dot_data,
			feature_names=iris.feature_names,
			class_names=iris.target_names,
			)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris_classification.pdf")

