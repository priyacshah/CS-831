# This script is for graphical user interface for Adult Cencus Income Prediction
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from turtle import left
from PIL import ImageTk, Image
from keras_preprocessing import image
from rsa import verify
import tensorflow as tf
import numpy as np
import requests
import urllib, io
import ssl
import pickle
import pandas as pd
import webbrowser
ssl._create_default_https_context = ssl._create_unverified_context

# Logo: https://www.fundingcircle.com/us/resources/balance-sheets-incomes-statements/

#start GUI
top=tk.Tk()
top.geometry('1400x1200')
top.title('Adult Census Income Predictor')
top.configure(background='#808080')
label=Label(top,background='#808080', font=('arial',10,'bold'))
sign_image = Label(top)

def classify_model_rf():
    # classifying given image using model 1
    model = pickle.load(open('model_rf.pkl', 'rb'))

    dataset = pd.read_csv('adult.csv')

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    dataset['income'] = le.fit_transform(dataset['income'])

    dataset = dataset.replace('?', np.nan)

    columns_with_nan = ['workclass', 'occupation', 'native.country']

    for col in columns_with_nan:
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            encoder = LabelEncoder()
            dataset[col] = encoder.fit_transform(dataset[col])
            
    X = dataset.drop('income', axis=1)
    Y = dataset['income']

    X = X.drop(['workclass', 'education', 'race', 'sex',
                'capital.loss', 'native.country', 'fnlwgt', 'relationship',
                'capital.gain'], axis=1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X = scaler.fit_transform(X) 

    education = edu.get()
    
    if education == "HS-grad":
        education_val = 9
    elif education == "Some-college":
        education_val = 10
    elif education == "7th-8th":
        education_val = 4
    elif education == "10th":
        education_val = 6
    elif education == "Doctorate":
        education_val = 16
    elif education == "Prof-school":
        education_val = 15
    elif education == "Bachelors":
        education_val = 13
    elif education == "Masters":
        education_val = 14
    elif education == "11th":
        education_val = 7
    elif education == "Assoc-acdm":
        education_val = 12
    elif education == "Assoc-voc":
        education_val = 11
    elif education == "1st-4th":
        education_val = 2
    elif education == "5th-6th":
        education_val = 3
    elif education == "12th":
        education_val = 8
    elif education == "9th":
        education_val = 5
    elif education == "Preschool":
        education_val = 1


    marital_name = variable.get()
    marital_value = 0
        
    if marital_name == 'Married-civ-spouse':
        marital_value = 1
    elif marital_name == 'Never-married':
        marital_value = 2
    elif marital_name == 'Divorced':
        marital_value = 3
    elif marital_name == 'Separated':
        marital_value = 4
    elif marital_name == 'Widowed':
        marital_value = 5
    elif marital_name == 'Married-spouse-absent':
        marital_value = 6
    elif marital_name == 'Married-AF-spouse':
        marital_value = 7

    occupation = variable1.get()

    if occupation == 'Prof-specialty':
        occupation_val = 9
    elif occupation == 'Exec-managerial':
        occupation_val = 3
    elif occupation == 'Machine-op-inspct':
        occupation_val = 6
    elif occupation == 'Other-service':
        occupation_val = 7
    elif occupation == 'Adm-clerical':
        occupation_val = 0
    elif occupation == 'Craft-repair':
        occupation_val = 2
    elif occupation == 'Transport-moving':
        occupation_val = 13
    elif occupation == 'Handlers-cleaners':
        occupation_val = 5
    elif occupation == 'Sales':
        occupation_val = 11
    elif occupation == 'Farming-fishing':
        occupation_val = 4
    elif occupation == 'Tech-support':
        occupation_val = 12
    elif occupation == 'Protective-serv':
        occupation_val = 10
    elif occupation == 'Armed-Forces':
        occupation_val = 1
    elif occupation == 'Priv-house-serv':
        occupation_val = 8


    age_value = int(E1.get())
    edu_num_value  = int(education_val)
    marital_value = int(marital_value)
    occupation_value = int(occupation_val)
    hours_value = int(H1.get())

    features = [age_value, edu_num_value, marital_value, 
                occupation_value, hours_value]

    int_features = [int(x) for x in features]
    final_features = [np.array(int_features)]
    prediction = model.predict(scaler.transform(final_features))
    print(prediction)
    if prediction == 1:
        output = "Income is more than $50K"

    if prediction == 0:
        output = "Income is less than $50K"
    print(output)
    label.configure(foreground='#011638', text=output, font=('arial',30, 'bold'))
    label.place(relx=0.69,rely=0.66)

def classify_model_svm():
    # classifying given image using model 1
    model = pickle.load(open('model_svm.pkl', 'rb'))

    dataset = pd.read_csv('adult.csv')

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()

    dataset['income'] = le.fit_transform(dataset['income'])

    dataset = dataset.replace('?', np.nan)

    columns_with_nan = ['workclass', 'occupation', 'native.country']

    for col in columns_with_nan:
        dataset[col].fillna(dataset[col].mode()[0], inplace=True)

    for col in dataset.columns:
        if dataset[col].dtypes == 'object':
            encoder = LabelEncoder()
            dataset[col] = encoder.fit_transform(dataset[col])
            
    X = dataset.drop('income', axis=1)
    Y = dataset['income']

    X = X.drop(['workclass', 'education', 'race', 'sex',
                'capital.loss', 'native.country', 'fnlwgt', 'relationship',
                'capital.gain'], axis=1)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    education = edu.get()
    
    if education == "HS-grad":
        education_val = 9
    elif education == "Some-college":
        education_val = 10
    elif education == "7th-8th":
        education_val = 4
    elif education == "10th":
        education_val = 6
    elif education == "Doctorate":
        education_val = 16
    elif education == "Prof-school":
        education_val = 15
    elif education == "Bachelors":
        education_val = 13
    elif education == "Masters":
        education_val = 14
    elif education == "11th":
        education_val = 7
    elif education == "Assoc-acdm":
        education_val = 12
    elif education == "Assoc-voc":
        education_val = 11
    elif education == "1st-4th":
        education_val = 2
    elif education == "5th-6th":
        education_val = 3
    elif education == "12th":
        education_val = 8
    elif education == "9th":
        education_val = 5
    elif education == "Preschool":
        education_val = 1

    marital_name = variable.get()
    marital_value = 0
        
    if marital_name == 'Married-civ-spouse':
        marital_value = 1
    elif marital_name == 'Never-married':
        marital_value = 2
    elif marital_name == 'Divorced':
        marital_value = 3
    elif marital_name == 'Separated':
        marital_value = 4
    elif marital_name == 'Widowed':
        marital_value = 5
    elif marital_name == 'Married-spouse-absent':
        marital_value = 6
    elif marital_name == 'Married-AF-spouse':
        marital_value = 7

    occupation = variable1.get()

    if occupation == 'Prof-specialty':
        occupation_val = 9
    elif occupation == 'Exec-managerial':
        occupation_val = 3
    elif occupation == 'Machine-op-inspct':
        occupation_val = 6
    elif occupation == 'Other-service':
        occupation_val = 7
    elif occupation == 'Adm-clerical':
        occupation_val = 0
    elif occupation == 'Craft-repair':
        occupation_val = 2
    elif occupation == 'Transport-moving':
        occupation_val = 13
    elif occupation == 'Handlers-cleaners':
        occupation_val = 5
    elif occupation == 'Sales':
        occupation_val = 11
    elif occupation == 'Farming-fishing':
        occupation_val = 4
    elif occupation == 'Tech-support':
        occupation_val = 12
    elif occupation == 'Protective-serv':
        occupation_val = 10
    elif occupation == 'Armed-Forces':
        occupation_val = 1
    elif occupation == 'Priv-house-serv':
        occupation_val = 8


    age_value = int(E1.get())
    edu_num_value  = int(education_val)
    marital_value = int(marital_value)
    occupation_value = int(occupation_val)
    hours_value = int(H1.get())

    features = [age_value, edu_num_value, marital_value, 
                occupation_value, hours_value]

    int_features = [int(x) for x in features]
    final_features = [np.array(int_features)]
    prediction = model.predict(scaler.transform(final_features))
    print(prediction)
    if prediction == 1:
        output = "Income is more than $50K"

    if prediction == 0:
        output = "Income is less than $50K"
    print(output)
    label.configure(foreground='#011638', text=output, font=('arial',30, 'bold'))
    label.place(relx=0.69,rely=0.66)


def show_classify_button_2():
    classify_b=Button(top,text="Classify using RF",
    command=lambda: classify_model_rf(),padx=10,pady=10)
    classify_b.configure(background='#1f0c38', foreground='black', font=('arial',20,'bold'))
    classify_b.place(relx=0.69,rely=0.56)

def show_classify_button_1():
    # classify_b=Button(top,text="Classify Image Using model 1",
    # command=lambda: classify_model_1(file_path),padx=10,pady=10)
    # classify_b.configure(background='#1f0c38', foreground='black', font=('arial',20,'bold'))
    # classify_b.place(relx=0.69,rely=0.46)
    classify_b=Button(top,text="Classify using SVM",
    command=lambda: classify_model_svm(),padx=10,pady=10)
    classify_b.configure(background='#1f0c38', foreground='black', font=('arial',20,'bold'))
    classify_b.place(relx=0.69,rely=0.46)

# inserting the logo in gui
img= (Image.open("logo.jpeg"))
resized_image= img.resize((220,220), Image.ANTIALIAS)
new_image= ImageTk.PhotoImage(resized_image)
logo_img = Label(image = new_image, borderwidth=0)
logo_img.place(x=-50,y=10)
logo_img.pack(pady=20)

show_classify_button_1()
show_classify_button_2()
# heading
heading = Label(top, text="Adult Census Income Predictor",pady=20, font=('arial',30,'bold'))
heading.configure(background='#808080',foreground='#000000')
heading.pack()

# Age
age = Label(top,text="Age:", font=('arial',25,'bold'), background="#808080",foreground="#000000")
age.pack()
E1 = tk.Entry(background="#FFFFFF", font=('arial'),foreground='#000')
E1.pack(pady=5)
# E1.pack(pady=10)

# years of education
years = Label(top,text="Years of Education:", font=('arial',25,'bold'), background="#808080",foreground="#000000")
years.pack()
edu = StringVar(top)
edu.set("Choose education") # default value
y1 = OptionMenu(top, edu,'HS-grad', 'Some-college', '7th-8th', '10th', 'Doctorate', 'Prof-school',
 'Bachelors', 'Masters', '11th', 'Assoc-acdm', 'Assoc-voc', '1st-4th', '5th-6th',
 '12th', '9th', 'Preschool')
y1.pack(pady=5)

# Maritial status
maritial = Label(top,text="Maritial Status", font=('arial',25,'bold'),background="#808080",foreground="#000000")
maritial.pack()
variable = StringVar(top)
variable.set("Choose status") # default value
w = OptionMenu(top, variable, "Married-civ-spouse", "Never-married", "Divorced","Separated","Widowed","Married-spouse-absent","Married-AF-spouse")
w.pack(pady=5)
# w.pack(pady=10)

# Occupation code
occupation = Label(top,text="Occupation code", font=('arial',25,'bold'),background="#808080",foreground="#000000")
occupation.pack()
variable1 = StringVar(top)
variable1.set("Choose occupation") # default value
O1 = OptionMenu(top, variable1, 'Prof-specialty', 'Exec-managerial', 'Machine-op-inspct', 'Other-service',
 'Adm-clerical', 'Craft-repair', 'Transport-moving', 'Handlers-cleaners',
 'Sales', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Armed-Forces',
 'Priv-house-serv')
O1.pack(pady=5)


# Hours of work per week
occupation = Label(top,text="Hours of Work/Week", font=('arial',25,'bold'),background="#808080",foreground="#000000")
occupation.pack()
H1 = Entry(top,background="#FFFFFF", font=('arial'),foreground='#000')
H1.pack()
H1.pack(pady=5)

top.mainloop()