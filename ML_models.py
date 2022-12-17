
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import preprocessing
import xgboost as xgb
from sklearn.metrics import classification_report
from sklearn.model_selection import learning_curve
import seaborn as sns

plt.switch_backend('agg')




def evaluate(model, X_train, y_train, X_test,  y_test, name="model", printing=False):


    scoring = "f1" #f1

    N, train_score, val_score = learning_curve(
        model, X_train, y_train, cv=4,
        scoring= scoring,  # neg_mean_squared_error  #f1 #accuracy
        train_sizes=np.linspace(.1, 1, 30)
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if printing:
        plt.figure()
        plt.title(name)
        plt.plot(N, train_score.mean(axis=1), label="train_score")
        plt.plot( N, val_score.mean(axis=1), label="val_score")
        plt.legend()

        plt.savefig(name+".png")
        print("saving done")     

    return y_pred

        

def create_datasets(csv_paths, variable):

    li = []
    for f in csv_paths :
        df = pd.read_csv(f)
        li.append(df)
    dataset = pd.concat(li, axis=0, ignore_index=True)

    print(dataset.shape)
    le = preprocessing.LabelEncoder()

    labels = np.array(dataset[variable] )
    labels = le.fit_transform(labels)
    features = dataset[[k for k in dataset.columns if k.isdigit()]]
    features = np.array(features)

    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=12)

    return X_train, y_train, X_test, y_test

    


def main():
    np.random.seed(333)

    # Create training and validation datasets
    train_csv_path = "outputs/UNet/final/UNet_5b4f_TCGA/autoencoding/features/concat_train.csv"
    val_csv_path = "outputs/UNet/final/UNet_5b4f_TCGA/autoencoding/features/concat_val.csv"
    test_csv_path = "outputs/UNet/final/UNet_5b4f_TCGA/autoencoding/features/concat_test.csv"
    csv_paths = [train_csv_path, val_csv_path, test_csv_path]
    X_train, y_train, X_test, y_test = create_datasets(csv_paths, "grade_binary")


    print(X_train.shape)
    print(X_test.shape)
    # Model initialization
    # KNN = KNeighborsClassifier(n_neighbors=5)
    XGB = xgb.XGBClassifier()

    dict_models = {"KNN":XGB}

    predictions = []
    for name, model in dict_models.items():
        predictions.append(evaluate(model, X_train, y_train,
                        X_test, y_test, name, printing=True))


    for m, pred in zip(dict_models.keys(), predictions):
        print(m)
        cm = classification_report(y_test, pred)
        print(cm)

if __name__ == '__main__':
    main()

