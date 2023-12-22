import os
import argparse 
import itertools

import numpy as np
from scipy.io import wavfile 
from hmmlearn import hmm
from sklearn.model_selection import train_test_split
from python_speech_features import mfcc, logfbank
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import tensorflow as tf





def neural_network_model(data):
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(X[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases':tf.Variable(tf.random_normal([n_classes])),}


    l1 = tf.add(tf.matmul(data,hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1,hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2,hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3,output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    # OLD VERSION:
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    hm_epochs = 10
    with tf.Session() as sess:
        # OLD:
        #sess.run(tf.initialize_all_variables())
        # NEW:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0 
            while i < len(X):
                start = i
                end = i + batch_size

                batch_x = np.array(X[start:end])
                batch_y = np.array(y_train[start:end])
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size

            print('Epoch', epoch, 'completed out of',hm_epochs,'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x:x_test, y:y_test}))





# Function to parse input arguments
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Trains the HMM classifier')
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Input folder containing the audio files in subfolders")
    return parser



if __name__=='__main__':
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    # Initialize variables
    X = np.array([])
    y_words = []
    # Parse the input directory
    for dirname in os.listdir(input_folder):
        # Get the name of the subfolder 
        subfolder = os.path.join(input_folder, dirname)
        #print(subfolder)

        if not os.path.isdir(subfolder): 
            continue

        # Extract the label
        label = subfolder[subfolder.rfind('\\') + 1:]

        # Iterate through the audio files 
        # for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]:
            # Read the input file
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)
            
            # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)
            flattened = mfcc_features.flatten()
            padded = np.pad(flattened,(0,1500-len(flattened)), mode='constant', constant_values=0)

            # Append to the variable X
            if len(X) == 0:
                X = padded
            else:
                X = np.vstack((X, padded))
            
            # Append the label
            y_words.append(label)



    print ("done Extracting features")

    lb = preprocessing.LabelBinarizer()
    lb.fit(['apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple'])
    y_train = lb.transform(y_words)



    # Test files
    test_folder = "D:\CSE\Thesis\Test Data"
    y_label_true = []
    y_test = []
    x_test = np.array([])

    for dirname in os.listdir(test_folder):
        subfolder = os.path.join(test_folder, dirname)

        if not os.path.isdir(subfolder): 
            continue

        #for each subfolder
        true_label = subfolder[subfolder.rfind('\\') + 1:]
    

        for filename in [x for x in os.listdir(subfolder) if x.endswith('.wav')][:]:
            y_label_true.append(true_label)
            filepath = os.path.join(subfolder, filename)
            sampling_freq, audio = wavfile.read(filepath)

    #         # Extract MFCC features
            mfcc_features = mfcc(audio, sampling_freq)
            flattened = mfcc_features.flatten()
            padded = np.pad(flattened,(0,1500-len(flattened)), mode='constant', constant_values=0)

            if len(x_test) == 0:
                x_test = padded
            else:
                x_test = np.vstack((x_test, padded))


    lb = preprocessing.LabelBinarizer()
    lb.fit(['apple', 'banana', 'kiwi', 'lime', 'orange', 'peach', 'pineapple'])
    y_test = lb.transform(y_label_true)

    n_nodes_hl1 = 500
    n_nodes_hl2 = 500
    n_nodes_hl3 = 500

    n_classes = 7
    batch_size = 100

    x = tf.placeholder('float', [None, len(X[0])])
    y = tf.placeholder('float')



    train_neural_network(x)


