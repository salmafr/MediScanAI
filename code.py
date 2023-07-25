import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tkinter as tk
from tkinter import *



#load_data
df= pd.read_csv('H:\YaneCode\Requirement Gathering and Analysis\Symptoms_Disease.csv')

#replace the values of column:prognosis by numerical values
df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

symptoms=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']
prognosis=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']
l=[0]*len(symptoms)
X=df[symptoms]
y=df[["prognosis"]]
X_train, X_test, y_train, y_test =train_test_split(X ,y ,test_size=0.3)

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
        t1.insert(END, disease[a])
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
        t2.insert(END, disease[a])
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
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==symptoms[k]):
                l[k]=1
    inputtest = [l]
    predict = gnb.predict(inputtest)
    predicted=predict[0]
    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break
    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")



#GUI
root = tk.Tk()
root.configure(background='white')
Symptom1 = tk.StringVar()
Symptom1.set("Select Here")
Symptom2 = tk.StringVar()
Symptom2.set("Select Here")
Symptom3 = tk.StringVar()
Symptom3.set("Select Here")
Symptom4 = tk.StringVar()
Symptom4.set("Select Here")
Symptom5 = tk.StringVar()
Symptom5.set("Select Here")
Name = tk.StringVar()
w2 = tk.Label(root, justify=CENTER, text="Disease Predictor using Machine Learning", fg="Red", bg="White")
w2.config(font=("Times",30,"bold italic"))
w2.grid(row=1, column=0, columnspan=2, padx=100)
w2.config(font=("Times",30,"bold italic"))
w2.grid(row=2, column=0, columnspan=2, padx=100)
NameLb = tk.Label(root, text="Name of the Patient", fg="black", bg="White")
NameLb.config(font=("Times",15,"bold italic"))
NameLb.grid(row=6, column=0, pady=15, sticky='W')
S1Lb = tk.Label(root, text="Symptom 1", fg="black", bg="White")
S1Lb.config(font=("Times",15,"bold italic"))
S1Lb.grid(row=7, column=0, pady=10, sticky='W')
S2Lb = tk.Label(root, text="Symptom 2", fg="black", bg="White")
S2Lb.config(font=("Times",15,"bold italic"))
S2Lb.grid(row=8, column=0, pady=10, sticky='W')
S3Lb = tk.Label(root, text="Symptom 3", fg="black", bg="White")
S3Lb.config(font=("Times",15,"bold italic"))
S3Lb.grid(row=9, column=0, pady=10, sticky='W')
S4Lb = tk.Label(root, text="Symptom 4", fg="black", bg="White")
S4Lb.config(font=("Times",15,"bold italic"))
S4Lb.grid(row=10, column=0, pady=10, sticky='W')
S5Lb = tk.Label(root, text="Symptom 5", fg="black", bg="White")
S5Lb.config(font=("Times",15,"bold italic"))
S5Lb.grid(row=11, column=0, pady=10, sticky='W')
lrLb = tk.Label(root, text="DecisionTree", fg="Red", bg="White")
lrLb.config(font=("Times",15,"bold italic"))
lrLb.grid(row=15, column=0, pady=10,sticky='W')
destreeLb = tk.Label(root, text="RandomForest", fg="Red", bg="white")
destreeLb.config(font=("Times",15,"bold italic"))
destreeLb.grid(row=17, column=0, pady=10, sticky='W')
ranfLb = tk.Label(root, text="NaiveBayes", fg="Red", bg="White")
ranfLb.config(font=("Times",15,"bold italic"))
ranfLb.grid(row=19, column=0, pady=10, sticky='W')
OPTIONS = sorted(symptoms)
NameEn = Entry(root, textvariable=Name)
NameEn.grid(row=6, column=1)
S1 = OptionMenu(root, Symptom1,*OPTIONS)
S1.grid(row=7, column=1)
S2 = OptionMenu(root, Symptom2,*OPTIONS)
S2.grid(row=8, column=1)
S3 = OptionMenu(root, Symptom3,*OPTIONS)
S3.grid(row=9, column=1)
S4 = OptionMenu(root, Symptom4,*OPTIONS)
S4.grid(row=10, column=1)
S5 = OptionMenu(root, Symptom5,*OPTIONS)
S5.grid(row=11, column=1)
dst = Button(root, text="Prediction 1", command=DecisionTree,bg="Red",fg="yellow")
dst.config(font=("Times",15,"bold italic"))
dst.grid(row=8, column=3,padx=10)
rnf = Button(root, text="Prediction 2", command=randomforest,bg="White",fg="green")
rnf.config(font=("Times",15,"bold italic"))
rnf.grid(row=9, column=3,padx=10)
lr = Button(root, text="Prediction 3", command=NaiveBayes,bg="Blue",fg="white")
lr.config(font=("Times",15,"bold italic"))
lr.grid(row=10, column=3,padx=10)
t1 = Text(root, height=1, width=40,bg="Light green",fg="red")
t1.config(font=("Times",15,"bold italic"))
t1.grid(row=15, column=1, padx=10)
t2 = Text(root, height=1, width=40,bg="White",fg="Blue")
t2.config(font=("Times",15,"bold italic"))
t2.grid(row=17, column=1 , padx=10)
t3 = Text(root, height=1, width=40,bg="red",fg="white")
t3.config(font=("Times",15,"bold italic"))
t3.grid(row=19, column=1 , padx=10)
root.mainloop()
