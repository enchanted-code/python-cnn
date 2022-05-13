from pathlib import Path
from urllib.request import urlopen

import arff
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd

DATA_SET_URL = "https://www.openml.org/data/download/1592290/phpgNaXZe"
ARFF_FILE_NAME = "dataset.arff"
CSV_FILE_NAME = "dataset.csv"
CSV_HEADER = "sbp,tobacco,ldl,adiposity,famhist,type,obesity,alcohol,age,chd"


def download_dataset(url, file_name):
    with urlopen(url) as response:
        content = response.read()
        with open(file_name, "wb") as fo:
            fo.write(content.strip())


def arff_to_csv(arff_data, header, file_name):
    with open(file_name, "wt") as fo:
        fo.write(header + "\n")
        for row in arff_data:
            fo.write(",".join([str(col) for col in row]) + "\n")


def main():
    if not Path(ARFF_FILE_NAME).exists():
        download_dataset(DATA_SET_URL, ARFF_FILE_NAME)

    if not Path(CSV_FILE_NAME).exists():
        with open(ARFF_FILE_NAME) as fo:
            arff_dataset = arff.load(fo)
            data = arff_dataset["data"]
            arff_to_csv(data, CSV_HEADER, CSV_FILE_NAME)

    df = pd.read_csv(CSV_FILE_NAME)

    x = pd.get_dummies(df.drop(['chd'], axis=1))
    y = df["chd"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2)


    model = Sequential()
    model.add(Dense(units=2, activation='relu', input_dim=len(x_train.columns)))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='sgd', metrics='accuracy')


    model.fit(x_train, y_train, epochs=20, batch_size=32, use_multiprocessing=True)

    y_predication = model.predict(x_test)
    y_predication = [0 if val < 0.5 else 1 for val in y_predication]


    accuracy = accuracy_score(y_test, y_predication)

    print(round(accuracy * 100, 4), "% Accurate")


    confusion = confusion_matrix(y_test, y_predication)

    ax = sns.heatmap(confusion, annot=True, cmap='Blues')
    ax.set_title("Confusion Matrix")
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ');
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    plt.show()


if __name__ == "__main__":
    # you might need to use this:
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    main()
