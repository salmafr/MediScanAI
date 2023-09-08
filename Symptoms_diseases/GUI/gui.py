import tkinter as tk
from tkinter import *
from algorithms import decision_tree, random_forest, naive_bayes
from data_loader import load_data

X_train, X_test, y_train, y_test, symptoms, prognosis, l = load_data()


root = tk.Tk()
root.configure(background='white')
def update_t1(result):
    t1.delete("1.0", END)
    t1.insert(END, result)

def update_t2(result):
    t2.delete("1.0", END)
    t2.insert(END, result)

def update_t3(result):
    t3.delete("1.0", END)
    t3.insert(END, result)

def on_decision_tree_click():
    result = decision_tree(X_train, X_test, y_train, y_test, symptoms, prognosis, l, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, t1)
    update_t1(result)

def on_random_forest_click():
    result = random_forest(X_train, X_test, y_train, y_test, symptoms, prognosis, l, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, t2)
    update_t2(result)

def on_naive_bayes_click():
    result = naive_bayes(X_train, X_test, y_train, y_test, symptoms, prognosis, l, Symptom1, Symptom2, Symptom3, Symptom4, Symptom5, t3)
    update_t3(result)

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
dst = Button(root, text="Prediction 1", command=on_decision_tree_click,bg="Red",fg="yellow")
dst.config(font=("Times",15,"bold italic"))
dst.grid(row=8, column=3,padx=10)
rnf = Button(root, text="Prediction 2", command=on_random_forest_click,bg="White",fg="green")
rnf.config(font=("Times",15,"bold italic"))
rnf.grid(row=9, column=3,padx=10)
lr = Button(root, text="Prediction 3", command=on_naive_bayes_click,bg="Blue",fg="white")
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
