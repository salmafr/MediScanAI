import numpy as np
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from data_loader import load_data


X_train, X_test, y_train, y_test, symptoms, prognosis, l = load_data()



#Decision tree
def decision_tree(X_train, X_test, y_train, y_test, symptoms, prognosis, l, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, t1):

    from sklearn import tree
    model1 = tree.DecisionTreeClassifier() 
    model1 = model1.fit(X_train,y_train)

    #calculating accuracy
    from sklearn.metrics import accuracy_score
    y_pred=model1.predict(X_test)
    print("Decision tree Accuracy:", accuracy_score(y_test, y_pred))


    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(symptoms)):
        for z in psymptoms:
            if(z==symptoms[k]):
                l[k]=1
    inputtest = [l]
    predict = model1.predict(inputtest)
    predicted=predict[0]
    
    for a in range(0,len(prognosis)):
        if(predicted == a):
            return prognosis[a]
    return "Not Found"
#randomforest
def random_forest(X_train, X_test, y_train, y_test, symptoms, prognosis, l, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, t2):

    from sklearn.ensemble import RandomForestClassifier
    model2 = RandomForestClassifier()
    model2 = model2.fit(X_train,np.ravel(y_train))

    # calculating accuracy 
    from sklearn.metrics import accuracy_score
    y_pred=model2.predict(X_test)
    print("RandomForest Accuracy:", accuracy_score(y_test, y_pred))
    
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(symptoms)):
        for z in psymptoms:
            if(z==symptoms[k]):
                l[k]=1
    inputtest = [l]
    predict = model2.predict(inputtest)
    predicted=predict[0]
    
    for a in range(0,len(prognosis)):
        if(predicted == a):
            return prognosis[a]
    return "Not Found"

#Naive Bayes
def naive_bayes(X_train, X_test, y_train, y_test, symptoms, prognosis, l, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, t3):

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X_train,np.ravel(y_train))

    # calculating accuracy
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print("Naive Bayes Accuracy:", accuracy_score(y_test, y_pred))
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(symptoms)):
        for z in psymptoms:
            if(z==symptoms[k]):
                l[k]=1
    inputtest = [l]
    predict = gnb.predict(inputtest)
    predicted=predict[0]
    
    for a in range(0,len(prognosis)):
        if(predicted == a):
            return prognosis[a]
    return "Not Found"
