# %%
import os
from os.path import join
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, learning_curve



import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier



# os.chdir("/Users/tom/drive/3-RECHERCHE/0-projets/en_cours/LESLY/projet")
os.chdir("/home/tbarba/projects/MultiModalBrainSurvival")
OUTPUT_DIR = "./"




def preproc(df):

    factors = ["age","gender","hta","dlp", "diab", "smoke"]

    df['age'] = np.round(df['age'] / 365,0)

    df['age'] = df['age'] / df['age'].max()
    df['height'] = df['height'] / 100
    df['bmi'] =  df['weight'] / df['height']**2 > 25
    df['bmi'] =  df['bmi'].astype(int)
    df['gender'] = df['gender'] - 1
    df['dlp'] = (df['cholesterol'] >1).astype(int)
    df['hta'] = np.logical_or(df['ap_hi'] > 140 , df['ap_lo'] > 80).astype(int)
    df['diab']  = (df['gluc'] > 1).astype(int)


    target = df['cardio'] 
    df = df[factors]



    return df, target

# %%


def evaluate(model, X_train, y_train, X_test, scoring, printing=False):

    N, train_score, val_score = learning_curve(
        model, X_train, y_train, cv=10,
        scoring=scoring,
        train_sizes=np.linspace(.1, 1, 30)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if printing:
        plt.figure()
        plt.title("cardio")
        plt.plot(N, train_score.mean(axis=1), label="train_score")
        plt.plot(N, val_score.mean(axis=1), label="val_score")
        plt.legend()
        plt.ylabel(scoring)
        # plt.savefig(join(OUTPUT_DIR , "results.png"))

    return y_pred

def main():


    DATA = "./cardio_train.csv"

    df = pd.read_csv(DATA, sep=";")

    df, target = preproc(df)

    X_train, X_test, y_train, y_test = train_test_split(
        df, target, test_size=0.2, random_state=42)

    XGB = XGBClassifier(tree_method='gpu_hist', gpu_id=1)
    scoring = "f1_weighted"

    y_pred = evaluate(XGB, X_train, y_train,
                      X_test, scoring, printing=True)

    res = accuracy_score(y_test, y_pred)
    print(res)

    return df




if __name__ == "__main__":
    y_pred = main()
# %%
