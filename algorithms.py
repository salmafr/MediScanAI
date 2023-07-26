#Algorithms
#Decision tree
def DecisionTree():

    from sklearn import tree
    model1 = tree.DecisionTreeClassifier() 
    model1 = model1.fit(X_train,y_train)

    #calculating accuracy
    from sklearn.metrics import accuracy_score
    y_pred=model1.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(symptoms)):
        for z in psymptoms:
            if(z==symptoms[k]):
                l[k]=1
    inputtest = [l]
    predict = model1.predict(inputtest)
    predicted=predict[0]
    h='no'
    for a in range(0,len(prognosis)):
        if(predicted == a):
            h='yes'
            break
    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, prognosis[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")

#randomforest
def randomforest():

    from sklearn.ensemble import RandomForestClassifier
    model2 = RandomForestClassifier()
    model2 = model2.fit(X_train,np.ravel(y_train))

    # calculating accuracy 
    from sklearn.metrics import accuracy_score
    y_pred=model2.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(symptoms)):
        for z in psymptoms:
            if(z==symptoms[k]):
                l[k]=1
    inputtest = [l]
    predict = model2.predict(inputtest)
    predicted=predict[0]
    h='no'
    for a in range(0,len(prognosis)):
        if(predicted == a):
            h='yes'
            break
    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, symptoms[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")

#Naive Bayes
def NaiveBayes():

    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X_train,np.ravel(y_train))

    # calculating accuracy
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(symptoms)):
        for z in psymptoms:
            if(z==symptoms[k]):
                l[k]=1
    inputtest = [l]
    predict = gnb.predict(inputtest)
    predicted=predict[0]
    h='no'
    for a in range(0,len(symptoms)):
        if(predicted == a):
            h='yes'
            break
    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, symptoms[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")