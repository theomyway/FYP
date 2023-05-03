import re
import urllib

from PIL import Image
from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from keras.applications import InceptionV3
from keras.applications.convnext import preprocess_input, decode_predictions
from keras.preprocessing import image as krs_image
from tensorboard import data
from urllib3.util import current_time

from .models import Note
from . import db
from werkzeug.utils import secure_filename
import json
from flask import Flask, render_template, request, session, redirect, url_for, flash
from tensorflow.keras.models import load_model
import numpy as np
from flask import current_app
import cv2
import os
import datetime
import seaborn as sns
import io
import matplotlib as mpl
import base64
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC
from builtins import str
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_confusion_matrix
from sklearn.metrics import plot_precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.extmath import density
from sklearn.neural_network import MLPClassifier
from imutils import paths
import numpy as np
import cv2
import time
import json

import cache

import matplotlib.pyplot as plt
import mahotas
from sklearn.model_selection import train_test_split

from time import time

import random
from numpy import asarray, loadtxt, load, save

from flask import Blueprint, render_template, request, flash, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, login_required, logout_user, current_user
from flask import Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
from flask import Flask, request, render_template, send_from_directory
import sqlite3
from flask_cache_external_assets import CacheExternalAssets
from flask import Flask
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

print('Libraries successfully imported')
from flask_login import UserMixin
from sqlalchemy.sql import func

conn = sqlite3.connect('database.db')
c = conn.cursor()

# Check if table exists
c.execute('''SELECT count(name) FROM sqlite_master WHERE type='table' AND name='Image' ''')
if c.fetchone()[0] == 1:
    print('Table already exists.')
else:
    # Create table
    c.execute('''CREATE TABLE Image
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                 img TEXT NOT NULL,
                 user_id INTEGER,
                 FOREIGN KEY(user_id) REFERENCES user(id))''')
    print('Table created successfully.')

conn.commit()
conn.close()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
# Create Database if it doesnt exist

UPLOAD_FOLDER = os.path.join('static', 'uploads')
# -----------------------------------------------------------------

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("select * from Image")
    data = cur.fetchall()
    con.close()
    user_id = session.get('user_id')

    return render_template("index.html", user=current_user)


@views.route('/index.html')
def mainpage():
    return render_template('index.html')


@views.route('/home.html')
def index():
    return render_template('home.html')


@views.route('/contact.html')
def contact():
    return render_template('contact.html')


@views.route('/news.html')
def news():
    return render_template('news.html')


@views.route('/about.html')
def about():
    return render_template('about.html')


@views.route('/faqs.html')
def faqs():
    return render_template('faqs.html')


@views.route('/prevention.html')
def prevention():
    return render_template('prevention.html')


@views.route('/upload.html')
def upload():
    return render_template('upload.html')


@views.route('/UploadsHistory.html', methods=['GET', 'POST'])
def history():
    if request.method == 'POST':
        note = request.form.get('note')  # Gets the note from the HTML

        if len(note) < 1:
            flash('Note is too short!', category='error')
        else:
            new_note = Note(data=note, user_id=current_user.id)  # providing the schema for the note
            db.session.add(new_note)  # adding the note to the database
            db.session.commit()
            flash('Note added!', category='success')

    con = sqlite3.connect("database.db")
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    cur.execute("SELECT * FROM Image WHERE user_id=?", (current_user.id,))
    data = cur.fetchall()
    con.close()
    return render_template('UploadsHistory.html', data=data, user=current_user)


@views.route('/delete-note', methods=['POST'])
def delete_note():
    note = json.loads(request.data)  # this function expects a JSON from the INDEX.js file
    noteId = note['noteId']
    note = Note.query.get(noteId)
    if note:
        if note.user_id == current_user.id:
            db.session.delete(note)
            db.session.commit()
    flash('Note deleted!', category='success')

    return render_template('UploadsHistory.html', data=data, user=current_user)


@views.route('/delete_image/<int:id>')
@login_required
def delete_image(id):
    con = sqlite3.connect("database.db")
    cur = con.cursor()
    cur.execute("DELETE FROM Image WHERE id=? AND user_id=?", (id, current_user.id))
    con.commit()
    con.close()
    flash('Image deleted successfully!', category='success')
    return redirect(url_for('views.history'))


def is_valid_xray_image(filename):
    # Load the image with OpenCV
    img = cv2.imread(filename)

    # Check the file extension
    if not allowed_file(filename):
        return False

    # Check that the image has low brightness
    brightness = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean()
    if brightness > 200:
        return False
        # Check that the image has a valid size
    if img.shape[0] not in [256, 299, 1024] or img.shape[1] not in [256, 299, 1024]:
        return False

    # with open(filename, 'rb') as f:
    # if re.search(b'\xff\xd8\xff\xe0\x00\x10JFIF', f.read()) is not None:
    # return False

    return True


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@views.route('/uploaded_chest', methods=['POST', 'GET'])
def uploaded_chest():
    global image, file_path, filename, file
    if request.method == 'POST':

        file = request.files['file']
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(request.url)

    if file.filename == '':
        flash('No selected file', category='error')
        return redirect(request.url)
    if file.filename != '':
        filename = file.filename
        file_path = os.path.join(app.static_folder, 'uploads', file.filename)
        file.save(os.path.join(app.static_folder, 'uploads', filename))

    try:
        # Check that the file is a valid X-ray image
        if is_valid_xray_image(file_path):

            con = sqlite3.connect("database.db")
            cur = con.cursor()
            cur.execute("INSERT INTO Image (img, user_id) VALUES (?, ?)", (file.filename, current_user.id))
            con.commit()

            con.close()
            con = sqlite3.connect("database.db")
            con.row_factory = sqlite3.Row
            cur = con.cursor()

            cur.execute("SELECT * FROM Image WHERE user_id=?", (current_user.id,))
            cur.fetchall()
            con.close()

            image = cv2.imread(file_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Color change to RGB  (Pre-Processing technique#01)
            image = cv2.resize(image,
                               (224, 224))  # Resizing image according to algo (Pre-Processing technique#02)
            image = np.array(image) / 255  # Converting image into a numpy array (Pre-Processing technique#03)
            image = np.expand_dims(image, axis=0)

        else:
            # If the file is not a valid X-ray image, delete it and return an error message

            file.close()
            flash('Please upload a valid X-ray image!', category='error')
            return redirect(url_for('views.upload'))

    except Exception as e:
        # If an error occurs, delete the file and return an error message

        file.close()
        flash('Error in processing the image!', category='error')
        return redirect(url_for('views.upload'))

    # -----------------------------------------------------------------------------------------------------------------

    def extract_color_histogram(image):

        # then perform "in place" normalization in OpenCV, and return the flattened histogram as the feature vector
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # Color space conversion
        hist_h = cv2.calcHist([image], [0], None, [180], [0, 180])  # Histogram Calculation
        hist_s = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([image], [2], None, [256], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h)  # Normalization
        hist_s = cv2.normalize(hist_s, hist_s)
        hist_v = cv2.normalize(hist_v, hist_v)
        return np.concatenate([hist_h, hist_s, hist_v], axis=0).reshape(-1)  # Feature concatenation

    def fd_haralick(image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Grey scale conversion
        # compute the Haralick Texture feature vector
        haralick = mahotas.features.haralick(gray).mean(axis=0)
        # return the result
        return haralick

    def fd_tas(image):
        # convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # compute the Threshold Adjacency Statistics feature vector
        value = mahotas.features.tas(gray)
        return value

    # -----------------------------------------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------------------------
    # Loading np array to train data
    features = load('features.npy', allow_pickle=True)
    labels = load('labels.npy', allow_pickle=True)
    inception_chest = load_model('inception.h5')
    covid_labels = np.load('covid_labels.npy')
    noncovid_labels = np.load('noncovid_labels.npy')
    covid_images = np.load('covid_images.npy')
    noncovid_images = np.load('noncovid_images.npy')

    # Splitting dataset in 20&test and training
    (trainF, testF, trainFL, testFL) = train_test_split(features, labels, test_size=0.20, random_state=10)
    # ----------------------------------------------------RFC Initials
    imagePaths = list(paths.list_images("Images-processed"))
    rand = random.choices(imagePaths, k=4)
    for (i, rand) in enumerate(rand):
        img = cv2.imread(rand)
        hist_h = cv2.calcHist([img], [0], None, [180], [0, 180])
        hist_s = cv2.calcHist([img], [1], None, [256], [0, 256])
        hist_v = cv2.calcHist([img], [2], None, [256], [0, 256])
        hist_h = cv2.normalize(hist_h, hist_h)
        hist_s = cv2.normalize(hist_s, hist_s)
        hist_v = cv2.normalize(hist_v, hist_v)
        label = rand.split(os.path.sep)[-1].split("-")[0]

    # ----------------> Serverside code
    t0 = time()
    X = features
    y = labels
    X_train = trainF
    X_test = testF
    y_train = trainFL
    y_test = testFL
    sc = StandardScaler()
    sc.fit(X_train)
    X_train = sc.transform(X_train)
    X_test = sc.transform(X_test)
    # ----------------> Initializing Random Forest classifier

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # ------------> Initializing KNN Classifier

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    # -------------> Initializing SVM model

    svm_model = SVC(probability=False)
    svm = CalibratedClassifierCV(svm_model, cv=5)
    svm.fit(X_train, y_train)

    # -------------------------------------

    # -------------Input img
    train_time = time() - t0

    histogram_imp = []
    hara_imp = []
    tas_imp = []

    images = cv2.imread(file_path)

    histogram_imp.append(extract_color_histogram(images))
    tas_imp.append(fd_tas(images))
    hara_imp.append(fd_haralick(images))

    hara_imp = np.array(hara_imp)
    tas_imp = np.array(tas_imp)
    histogram_imp = np.array(histogram_imp)
    features_imp = np.concatenate((hara_imp, tas_imp), axis=1)
    features_imp = np.concatenate((features_imp, histogram_imp), axis=1)

    X_pred = features_imp
    X_pred = sc.transform(X_pred)
    # ----------------------------------------> Inceptionv3 Predeiction-----------------------

    inception_pred = inception_chest.predict(image)
    probability = inception_pred[0]
    print("Inception Predictions:")
    if probability[0] > 0.5:
        inception_chest_pred = str('%.2f' % (probability[0] * 100) + '% COVID Patient')
    else:
        inception_chest_pred = str('%.2f' % ((1 - probability[0]) * 100) + '% Non-COVID Patient')
    print(inception_chest_pred)
    # ---------------------------------Confusion Matrix Inceptionv3 <-------------------

    # Convert to array and Normalize to interval of [0,1]
    covid_images = np.array(covid_images) / 255
    noncovid_images = np.array(noncovid_images) / 255

    # split into training and testing
    covid_x_train, covid_x_test, covid_y_train, covid_y_test = train_test_split(covid_images, covid_labels,
                                                                                test_size=0.2)
    noncovid_x_train, noncovid_x_test, noncovid_y_train, noncovid_y_test = train_test_split(noncovid_images,
                                                                                            noncovid_labels,
                                                                                            test_size=0.2)

    # concatenate the training and testing data

    X_testCNN = np.concatenate((noncovid_x_test, covid_x_test), axis=0)
    y_testCNN = np.concatenate((noncovid_y_test, covid_y_test), axis=0)

    # reshape y_testCNN to create a column vector of labels
    y_testCNN = np.reshape(y_testCNN, (-1, 1))

    # make predictions on test set
    y_pred = inception_chest.predict(X_testCNN)

    # get labels for predictions
    y_pred_labels = np.argmax(y_pred, axis=1)
    y_test_labels = np.argmax(y_testCNN, axis=1)

    # calculate confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=['NON_COVID', 'COVID-19'],
           yticklabels=['NON_COVID', 'COVID-19'], title='Confusion matrix', ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), fontsize=22, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Convert the plot to a PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_VGG19 = base64.b64encode(img.getvalue()).decode()

    # ----------------------------------------------------------

    # ----------------------------------------->Random-Forest PREDICTION--------------------------------
    rfc_pred = rfc.predict_proba(X_pred)[0]

    # Extract the probability of the positive class (COVID-19) from the predictions
    probability = rfc_pred[0]

    # Print the Random Forest Prediction Score
    print("Random Forest Prediction Score:")
    print(probability)

    # Check if the predicted probability is greater than 0.5 to determine the predicted class
    if probability > 0.5:
        rfc_chest_pred = str('%.2f' % (probability * 100) + '% COVID Patient')
    else:
        rfc_chest_pred = str('%.2f' % ((1 - probability) * 100) + '% Non-COVID Patient')

    # Print the Random Forest Prediction
    print("Random Forest Prediction:")
    print(rfc_chest_pred)
    # ---------------------------------Confusion Matric RFC
    pred = rfc.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=['NON_COVID', 'COVID-19'],
           yticklabels=['NON_COVID', 'COVID-19'], title='Confusion matrix', ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), fontsize=22, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Convert the plot to a PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_rfc = base64.b64encode(img.getvalue()).decode()

    # --------------------------------------------------------------------------------------------------

    knn_pred = knn.predict_proba(X_pred)[0]
    probability = knn_pred[0]
    print("KNN Prediction Score:")
    print(probability)
    if probability > 0.5:
        knn_chest_pred = str('%.2f' % (probability * 100) + '% COVID Patient')
    else:
        knn_chest_pred = str('%.2f' % ((1 - probability) * 100) + '% Non-COVID Patient')
    print("KNN score:")
    print(knn_chest_pred)

    # -----------------------------------------------------Plotting confusion matrix for KNN--------------
    pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=['NON_COVID', 'COVID-19'],
           yticklabels=['NON_COVID', 'COVID-19'], title='Confusion matrix', ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), fontsize=22, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Convert the plot to a PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_knn = base64.b64encode(img.getvalue()).decode()
    # --------------------------------------------------------------------------------------------------
    svm_pred = svm.predict_proba(X_pred)[0]

    probability = svm_pred[0]
    print("Random Forest Prediction Score:")
    print(probability)
    if probability > 0.5:
        svm_chest_pred = str('%.2f' % (probability * 100) + '% COVID Patient')
    else:
        svm_chest_pred = str('%.2f' % ((1 - probability) * 100) + '% Non-COVID Patient')
    print("SVM score:")
    print(svm_chest_pred)
    # -------------------------------->Confusion Matric Of SVM Model
    pred = svm.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]), xticklabels=['NON_COVID', 'COVID-19'],
           yticklabels=['NON_COVID', 'COVID-19'], title='Confusion matrix', ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=18)
    plt.setp(ax.get_yticklabels(), fontsize=18)
    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'), fontsize=22, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()

    # Convert the plot to a PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url_svm = base64.b64encode(img.getvalue()).decode()

    flash('Image uploaded successfully!', category='success')

    return render_template('results_chest.html', plot_url_knn=plot_url_knn, plot_url_rfc=plot_url_rfc,
                           plot_url_svm=plot_url_svm, rfc_chest_pred=rfc_chest_pred, knn_chest_pred=knn_chest_pred,
                           svm_chest_pred=svm_chest_pred, inception_chest_pred=inception_chest_pred,
                           plot_url_VGG19=plot_url_VGG19, filename=filename)
