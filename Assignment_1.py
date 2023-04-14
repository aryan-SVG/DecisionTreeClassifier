from sklearn import datasets
import matplotlib.pyplot as plt


dia = datasets.fetch_openml(data_id=1504)


dia1 = datasets.fetch_openml(data_id=971)

from sklearn import tree
mytree = tree.DecisionTreeClassifier(criterion="entropy")
mytree1 = tree.DecisionTreeClassifier(criterion="entropy")


mytree.fit(dia.data,dia.target)
mytree1.fit(dia1.data,dia1.target)


print(tree.export_text(mytree))




print(tree.export_text(mytree1))

predictions = mytree.predict(dia.data)
    

predictions1 = mytree1.predict(dia1.data)

predictions

predictions1

from sklearn import model_selection
def subtask1(dia):
    cv=[]
    train = []
    test = []
    min_leafs = [90+i for i in range(5)]
    for i in min_leafs:
        dtc = tree.DecisionTreeClassifier(min_samples_leaf=i)
        d = model_selection.cross_validate(dtc,dia.data, dia.target, scoring="roc_auc", cv=10,return_train_score=True)
        cv.append([d['train_score'].mean(),d['test_score'].mean()])
        train.append(d['train_score'].mean())
        test.append(d['test_score'].mean())
    max_ind = 0
    min_ind = 0
    for i in range(len(train)):
        t = train[max_ind] - test[max_ind]
        l = train[i] - test[i]
        if(t<l): max_ind = i
        if(train[min_ind]>train[i]): min_ind = i
    plt.axvline(x=min_leafs[max_ind] )#
    plt.axvline(x=min_leafs[min_ind])#
    plt.annotate(text="Overfit",xy=(min_leafs[max_ind],train[max_ind]))
    plt.annotate(text="Underfit",xy=(min_leafs[min_ind],train[min_ind]))
    
        
        
    plt.plot(min_leafs,train,color="red")
    plt.plot(min_leafs,test,color="black")
    
    plt.xlabel("Min sample Leaf")
    plt.ylabel("ROC AUC")
    plt.show()
    return cv

subtask1(dia)
subtask1(dia1)

parameters = [{"min_samples_leaf":[90+i for i in range(5)]}]

mytree = tree.DecisionTreeClassifier(criterion="entropy")
mytree1 = tree.DecisionTreeClassifier(criterion="entropy")

tuned_dtc = model_selection.GridSearchCV(mytree, parameters, scoring="roc_auc", cv=10,return_train_score=True)
tuned_dtc1 = model_selection.GridSearchCV(mytree1, parameters, scoring="roc_auc", cv=10,return_train_score=True)

tuned_dtc.fit(dia.data, dia.target)
tuned_dtc1.fit(dia1.data, dia1.target)

tuned_dtc.cv_results_.keys()


tuned_dtc.cv_results_["mean_test_score"]


#from sklearn import datasets, model_selection, tree

def subtask2(dia):
    parameters = [{"min_samples_leaf":[90+i for i in range(5)]}]
    mytree = tree.DecisionTreeClassifier(criterion="entropy")
    tuned_dtc = model_selection.GridSearchCV(mytree, parameters, scoring="roc_auc", cv=10, return_train_score=True)
    tuned_dtc.fit(dia.data, dia.target)
    
    min_sample_leaf = [90+i for i in range(5)]
    plt.plot(min_sample_leaf, tuned_dtc.cv_results_["mean_test_score"])
    plt.plot(min_sample_leaf, tuned_dtc.cv_results_["mean_train_score"])
    plt.xlabel("Min sample Leaf")
    plt.ylabel("ROC AUC")
    plt.annotate(text="Best parameter", xy=[tuned_dtc.best_params_["min_samples_leaf"], tuned_dtc.cv_results_["mean_train_score"][min_sample_leaf.index(tuned_dtc.best_params_["min_samples_leaf"])]])
    plt.annotate(text="Best parameter", xy=[tuned_dtc.best_params_["min_samples_leaf"], tuned_dtc.cv_results_["mean_test_score"][min_sample_leaf.index(tuned_dtc.best_params_["min_samples_leaf"])]])
    plt.axvline(x=tuned_dtc.best_params_["min_samples_leaf"])
    plt.show()
    
    
subtask2(dia)
subtask2(dia1)
