from sklearn import tree
 # scikitlearn module/library
 # tree is the classifier(box of rules) used 

# training data qunatified(ints instead of strings-say xyz=1 and mno=2)
features =[[150,1,2],[170,0,2],[130,1,1],[180,0,1]]
labels = [0,1,0,1]

classifier=tree.DecisionTreeClassifier() # empty classifier

# now we must train the classifier through learning
# we learn by finding patterns in our training data
classifier = classifier.fit(features, labels)

#use predict to classify the new feature/data
print(classifier.predict([[190,0,0]]))


