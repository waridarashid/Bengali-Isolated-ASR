import os
import argparse 
import itertools

import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
#from features import mfcc
from python_speech_features import mfcc, logfbank
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# Class to handle all HMM related processing
class HMMTrainer(object):
    def __init__(self, model_name='GaussianHMM', n_components=4, cov_type='diag', n_iter=1000):
        self.model_name = model_name
        self.n_components = n_components
        self.cov_type = cov_type
        self.n_iter = n_iter
        self.models = []

        if self.model_name == 'GaussianHMM':
            self.model = hmm.GaussianHMM(n_components=self.n_components, 
                    covariance_type=self.cov_type, n_iter=self.n_iter)
        else:
            raise TypeError('Invalid model type')

    # X is a 2D numpy array where each row is 13D
    def train(self, X):
        np.seterr(all='ignore')
        self.models.append(self.model.fit(X))

    # Run the model on input data
    def get_score(self, input_data):
        return self.model.score(input_data)

if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    hmm_models = []

    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder 
        subfolder = os.path.join(input_folder, dirname)
        #print(subfolder)

        if not os.path.isdir(subfolder): 
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('\\') + 1:]

        # Initialize variables
        X = np.array([])
        y_words = []

        # Iterate through the audio files (leaving 1 file for testing in each class)
        # for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-2]:
            # Read the input file
            #print (filename)
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)
            #print ('\nMFCC:\nNumber of windows =', mfcc_features.shape[0], '\nFilename: ', filename)
            #print ('Length of each feature =', mfcc_features.shape[1])

            # Append to the variable X
            if len(X) == 0:
                X = mfcc_features
            else:
                X = np.append(X, mfcc_features, axis=0)
            
            # Append the label
            y_words.append(label)


        # Train and save HMM model
        hmm_trainer = HMMTrainer()
        hmm_trainer.train(X)
        hmm_models.append((hmm_trainer, label))
        hmm_trainer = None



    print ("done training")

    #modified test files
    test_folder = "D:\CSE\Thesis\Test Data"
    y_label_true = []
    y_label_predicted = []

    for dirname in os.listdir(test_folder):
        subfolder = os.path.join(test_folder, dirname)

        if not os.path.isdir(subfolder): 
            continue

        #for each subfolder
        true_label = subfolder[subfolder.rfind('\\') + 1:]
        # print ("true_label: ", true_label)

        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:]:
            y_label_true.append(true_label)
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)

            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)
            max_score = float('-inf')
            predicted_label = None

            #iterate through all HMM models and pick the one with highest score
            for item in hmm_models:
                hmm_model, label = item
                score = hmm_model.get_score(mfcc_features)
                # print (score)
                if score > max_score:
                    max_score = score
                    predicted_label = label

            # print ("Predicted:", predicted_label) 
            y_label_predicted.append(predicted_label)


    print (y_label_true, "\n", y_label_predicted)

    # lb = preprocessing.LabelBinarizer()
    # lb.fit(['apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple'])
    # print(lb.classes_)
    # y_label_true_transformed = lb.transform(y_label_true)
    # y_label_predicted_transformed = lb.transform(y_label_predicted)


    # print ('TRUE ', y_label_true_transformed, "\n PREDICTED ", y_label_predicted_transformed)
    cnf_matrix = confusion_matrix(y_label_true, y_label_predicted, labels=["apple", "banana", "kiwi", "lime", "orange", "peach", "pineapple"])
    # print("CM: \n", cnf_matrix)

    # Plotting the confusion_matrix
    # Plot non-normalized confusion matrix
    class_names = ["apple", "banana", "kiwi", "lime", "orange", "peach", "pineapple"]
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix, without normalization')
    score = accuracy_score(y_label_true, y_label_predicted);
    print (score)
    # plt.show()