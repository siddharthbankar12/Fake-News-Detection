import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import re
import string
import pickle
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Inserting fake and real dataset
df_fake = pd.read_csv("datasets/Fake.csv")
df_true = pd.read_csv("datasets/True.csv")

# Inserting a column called "class" for fake and real news dataset to categorize fake and true news.
df_fake["class"] = 0
df_true["class"] = 1

# Removing last 10 rows from both the dataset, for manual testing
df_fake = df_fake.iloc[:-10]
df_true = df_true.iloc[:-10]

# Merging the main fake and true dataframe
df_marge = pd.concat([df_fake, df_true], axis=0)

# Dropping unnecessary columns
df = df_marge.drop(["title", "subject", "date"], axis=1)

# Randomly shuffling the dataframe
df = df.sample(frac=1).reset_index(drop=True)

# Preprocessing function
def wordopt(text):
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)
    return text

# Applying preprocessing
df["text"] = df["text"].apply(wordopt)

# Defining dependent and independent variables
x = df["text"]
y = df["class"]

# Splitting the dataset into training set and testing set.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# Convert text to vectors
vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

# Save TfidfVectorizer
with open("models/vectorizer.pkl", 'wb') as file:
    pickle.dump(vectorization, file)

# Model Training
LR = LogisticRegression()
DT = DecisionTreeClassifier()
GBC = GradientBoostingClassifier(random_state=0)
RFC = RandomForestClassifier(random_state=0)

models = [LR, DT, GBC, RFC]

for model in models:
    model.fit(xv_train, y_train)

# Saving models
folder_name = "models"
os.makedirs(folder_name, exist_ok=True)

for idx, model in enumerate(models):
    model_path = os.path.join(folder_name, f"model_{idx}.pkl")
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model {idx} saved successfully.")

print("All models saved successfully in the folder:", folder_name)
