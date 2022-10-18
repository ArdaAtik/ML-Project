from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score, accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import os

app = Flask(__name__)
dir_path = os.path.dirname(os.path.realpath(__file__))


def clear_data():
    music = pd.read_csv("music_genre.csv")
    duplicated = music.duplicated().any()
    print(duplicated)
    # checking for duplicates to delete
    music.drop([10000, 10001, 10002, 10003, 10004], inplace=True)
    # deleting $ string with dollar sign in them
    music['artist_name'] = music['artist_name'].str.replace('$', '', regex=True)
    music.reset_index(inplace=True)
    # deleting attribute with strings in them that can't be used as a prediction
    music = music.drop(["index", "instance_id", "track_name", "obtained_date"], axis=1)

    return music


def create_bar_plot(x, y, xlab, title, filename):
    plt.barh(x, y)
    plt.xlabel(xlab)
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def plot_artists(music):
    print(music[music["artist_name"] == "empty_field"])  # checking for music entries with no artist name
    artists = music["artist_name"].value_counts()[:20].sort_values(ascending=True)
    # plotting artists by their count
    ##########

    filename = f'{dir_path}/static/image.jpg'
    create_bar_plot(artists.index, artists, "Number of songs per artist", "Songs per artist", filename)
    # plotting top 20 artists by their count
    #########

    music = music.drop(music[music["artist_name"] == "empty_field"].index)  # removing empty fields
    top_20_artists = music["artist_name"].value_counts()[:20].sort_values(ascending=True)
    filename = f'{dir_path}/static/image2.jpg'
    create_bar_plot(top_20_artists.index, top_20_artists, "Number of songs per artist", "Top 20 artist", filename)
    music.drop("artist_name", axis=1, inplace=True)
    # dropping artist_names for the same reason we dropped ids and dates
    return music


def visualize_data(music):
    # count plot function that saves the plots as a jpg
    def plot_counts(feature, filenum, order=None):
        sns.countplot(x=feature, data=music, palette="ocean", order=order)
        plt.title(f"Counts in each {feature}")
        filename = f'{dir_path}/static/image{filenum}.jpg'
        plt.savefig(filename)

    # creating plots
    plot_counts("key", '3', ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"])
    plot_counts("mode", '4')
    plot_counts("music_genre", '5')

    # ----------------------------------------------#
    # finding entries where tempo == ? (to delete)
    music = music.drop(music[music["tempo"] == "?"].index)
    # because it can be used as a prediction we are turning tempo entries into float values
    music["tempo"] = music["tempo"].astype("float")
    music["tempo"] = np.around(music["tempo"], decimals=2)

    numeric_features = music.drop(["key", "music_genre", "mode"], axis=1)

    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(15, 15))
    fig.delaxes(axs[3][2])
    index = 0

    axs = axs.flatten()
    for k, v in numeric_features.items():
        sns.histplot(v, ax=axs[index])
        index += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.savefig(f'{dir_path}/static/image6.jpg')
    # histogram of every numeric value in the data frame

    fig, axs = plt.subplots(ncols=3, nrows=4, figsize=(15, 15))
    fig.delaxes(axs[3][2])
    idx = 0
    axs = axs.flatten()
    for k, v in numeric_features.items():
        sns.boxplot(y=k, data=numeric_features, ax=axs[idx])
        idx += 1
    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)
    plt.savefig(f'{dir_path}/static/image7.jpg')
    # boxplot of every numeric value in the data frame
    return music


def encode(music):
    # encoding string values to floats
    key_encoder = LabelEncoder()
    music["key"] = key_encoder.fit_transform(music["key"])

    mode_encoder = LabelEncoder()
    music["mode"] = mode_encoder.fit_transform(music["mode"])

    return music


def classification_task(estimator, features, labels):
    predictions = estimator.predict(features)
    # estimating the accuracy of the model
    print(f"Accuracy: {accuracy_score(labels, predictions)}")
    print(f"F1 score: {f1_score(labels, predictions, average='weighted')}")


def preprocess_classification(music):
    music_features = music.drop("music_genre", axis=1)
    music_labels = music["music_genre"]

    scaler = StandardScaler()  # checks standard deviation between of 0-1
    music_features_scaled = scaler.fit_transform(music_features)

    # creating test model , train model , and a verification model
    tr_val_f, test_features, tr_val_l, test_labels = train_test_split(
        music_features_scaled, music_labels, test_size=0.1, stratify=music_labels)

    train_features, val_features, train_labels, val_labels = train_test_split(
        tr_val_f, tr_val_l, test_size=len(test_labels), stratify=tr_val_l)

    f1 = make_scorer(f1_score, average="weighted")
    params = {
        "n_estimators": [10, 15, 20, 25, 30, 35],
        "max_depth": [5, 10, 15, 20, 25],
        "min_samples_leaf": [1, 2, 3, 4, 5]
    }
    #
    rfc = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rfc, param_grid=params, scoring=f1, cv=5)
    grid_search.fit(train_features, train_labels)
    model = RandomForestClassifier(n_estimators=35, max_depth=15, min_samples_leaf=4)
    model.fit(train_features, train_labels)

    # using the model to create predicted plot
    classification_task(model, train_features, train_labels)

    classification_task(model, val_features, val_labels)

    classification_task(model, test_features, test_labels)
    sns.heatmap(confusion_matrix(test_labels, model.predict(test_features)),
                annot=True,
                fmt=".0f",
                cmap="vlag",
                linewidths=2,
                linecolor="red",
                xticklabels=model.classes_,
                yticklabels=model.classes_)
    plt.title("Actual values")
    plt.ylabel("Predicted values")
    plt.tight_layout()
    plt.savefig(f'{dir_path}/static/image8.jpg')

    predicted_labels = model.predict_proba(test_features)
    try:
        roc_auc_score(test_labels, predicted_labels, multi_class="ovr")
    except ValueError:
        pass


@app.route('/')
def hello_world():  # put application's code here
    music = clear_data()
    music = plot_artists(music)
    music = visualize_data(music)
    music = encode(music)
    preprocess_classification(music)
    return render_template('index.html')


if __name__ == '__main__':
    app.run()
