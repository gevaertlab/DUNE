
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn import preprocessing
plt.switch_backend('agg')


def train_model():

    for epoch in range(10):
        pass
        

def create_datasets(list_of_paths, variable):

    le = preprocessing.LabelEncoder()

    final_df = pd.DataFrame()
    for path in list_of_paths:
        df = pd.read_csv(path)
        final_df = pd.concat([final_df, df])


    labels = np.array(final_df[variable] )
    labels = le.fit_transform(labels)
    features = final_df[[k for k in final_df.columns if k.isdigit()]]
    features = np.array(features)

    return features, labels

    

def main():
    np.random.seed(333)

    # Create training and validation datasets
    train_csv_path = "outputs/UNet/UNet_6b_4f_UKfull/autoencoding/features/concat_train.csv"
    val_csv_path = "outputs/UNet/UNet_6b_4f_UKfull/autoencoding/features/concat_val.csv"
    test_csv_path = "outputs/UNet/UNet_6b_4f_UKfull/autoencoding/features/concat_test.csv"
    
    X_train, y_train = create_datasets([train_csv_path, val_csv_path], variable="sex")
    X_test, y_test = create_datasets([test_csv_path], variable="sex")


    # Model initialization
    model = KNeighborsClassifier(n_neighbors=5)
    model = RandomForestClassifier()

    model.fit(X_train, y_train)
    y_pred=  model.predict(X_test)


    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


if __name__ == '__main__':
    main()

