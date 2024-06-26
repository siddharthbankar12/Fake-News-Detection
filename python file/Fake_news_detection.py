# Importing required library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string
import pickle

# Inserting fake and real dataset
df_fake = pd.read_csv("Fake.csv")
df_true = pd.read_csv("True.csv")

df_fake.head(5)
df_true.head(5)

# Inserting a column called "class" for fake and real news dataset to categories fake and true news. 
df_fake["class"] = 0
df_true["class"] = 1

# Removing last 10 rows from both the dataset, for manual testing  
df_fake.shape, df_true.shape

df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

df_fake.shape, df_true.shape

# Merging the manual testing dataframe in single dataset and save it in a csv file
df_fake_manual_testing.loc[:, "class"] = 0
df_true_manual_testing.loc[:, "class"] = 1

df_fake_manual_testing.head(10)
df_true_manual_testing.head(10)

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("manual_testing.csv")

# Merging the main fake and true dataframe
df_marge = pd.concat([df_fake, df_true], axis =0 )
df_marge.head(10)

df_marge.columns

# "title",  "subject" and "date" columns is not required for detecting the fake news, so I am going to drop the columns.
df = df_marge.drop(["title", "subject","date"], axis = 1)

df.isnull().sum()

# Randomly shuffling the dataframe 
df = df.sample(frac = 1)

df.head()

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

df.columns

df.head()

# Creating a function to convert the text in lowercase, remove the extra space, special chr., ulr and links.
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

df["text"] = df["text"].apply(wordopt)

# Defining dependent and independent variable as x and y
x = df["text"]
y = df["class"]

# Splitting the dataset into training set and testing set. 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)


# Convert text to vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)


# 1. Logistic Regression
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(xv_train,y_train)

pred_lr=LR.predict(xv_test)

LR.score(xv_test, y_test)

print("\n1. Logistic Regression Report:\n")
print(classification_report(y_test, pred_lr))
print("-" * 60)



# 2. Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier

DT = DecisionTreeClassifier()
DT.fit(xv_train, y_train)

pred_dt = DT.predict(xv_test)

DT.score(xv_test, y_test)

print("2. Decision Tree Report:\n")
print(classification_report(y_test, pred_dt))
print("-" * 60)



# 3. Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier(random_state=0)
GBC.fit(xv_train, y_train)

pred_gbc = GBC.predict(xv_test)

GBC.score(xv_test, y_test)

print("3. Gradient Boosting Report:\n")
print(classification_report(y_test, pred_gbc))
print("-" * 60)



# 4. Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(random_state=0)
RFC.fit(xv_train, y_train)

pred_rfc = RFC.predict(xv_test)

RFC.score(xv_test, y_test)


print("4. Random Forest Report:\n")
print(classification_report(y_test, pred_rfc))
print("-" * 60)



# Model Testing With Manual Entry
# News
def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"
    
def manual_testing(news):
    testing_news = {"text":[news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt) 
    new_x_test = new_def_test["text"]
    new_xv_test = vectorization.transform(new_x_test)
    pred_LR = LR.predict(new_xv_test)
    pred_DT = DT.predict(new_xv_test)
    pred_GBC = GBC.predict(new_xv_test)
    pred_RFC = RFC.predict(new_xv_test)

    return print("\n\nLR Prediction: {} \nDT Prediction: {} \nGBC Prediction: {} \nRFC Prediction: {}".format(output_lable(pred_LR[0]), output_lable(pred_DT[0]), output_lable(pred_GBC[0]), output_lable(pred_RFC[0])))


news = str(input())
manual_testing(news)