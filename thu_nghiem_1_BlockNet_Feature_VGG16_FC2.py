import os
import numpy as np
import pandas as pd
import time
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder, label_binarize
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
import pickle
from pprint import pprint
import joblib
import tensorflow as tf

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
# from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import RandomForestClassifier
from scikeras.wrappers import KerasClassifier
from keras.models import load_model
from numpy import dstack
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from datetime import datetime
from sklearn.metrics import roc_auc_score

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
# from transformers import TFAutoModel, AutoTokenizer
from datetime import datetime
from sklearn.utils import shuffle

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def create_keras_model_2(X_train, y_train):
    num_classes = len(np.unique(y_train))
    model = Sequential()
    model.add(Dense(512, input_dim=X_train.shape[1]))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def cal_sensitivity_specificity(conf_matrix, num_classes):
    sensitivity = []
    specificity = []

    for i in range(num_classes):
        TP = conf_matrix[i, i]
        FN = sum(conf_matrix[i, :]) - TP
        FP = sum(conf_matrix[:, i]) - TP
        TN = sum(sum(conf_matrix)) - TP - FN - FP

        sensitivity_i = TP / (TP + FN) if TP + FN > 0 else 0
        specificity_i = TN / (TN + FP) if TN + FP > 0 else 0

        sensitivity.append(sensitivity_i)
        specificity.append(specificity_i)

    return sensitivity, specificity


# Define a function to save Sensitivity and Specificity to a CSV file
def save_sensitivity_specificity(model, X_test, test_labels_encoded, result_out, model_name, layer_name, epoch):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert y_pred_stacked to multiclass format
    y_pred_multiclass = np.argmax(y_pred, axis=1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_labels_encoded, y_pred_multiclass)

    # Calculate sensitivity and specificity for each class
    num_classes = conf_matrix.shape[0]
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - tp - fn - fp

        sensitivity[i] = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity[i] = tn / (tn + fp) if tn + fp > 0 else 0.0

    # Create a DataFrame to store Sensitivity and Specificity data
    sensitivity_specificity_data = {
        'Class': list(range(num_classes)),
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }
    sensitivity_specificity_df = pd.DataFrame(sensitivity_specificity_data)

    # Save the DataFrame to a CSV file
    sensitivity_specificity_df.to_csv(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_sensitivity_specificity.csv'), index=False)

    
    
def plot_roc_curve_multilabel(test_labels_multilabel, y_pred, class_labels, result_out,model_name, layer_name, epoch):
    """
    Plot ROC curves for each class in a multilabel classification problem.
    
    Args:
        test_labels_multilabel (array-like): True labels in multilabel format.
        y_pred_stack (array-like): Predicted labels.
        class_labels (array-like): List of class labels.
        result_out (str): Path to save the ROC curve plot.
    """
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    plt.figure(figsize=(8, 6))
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(test_labels_multilabel[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        linestyle = '--' if i % 2 == 0 else '-'  # Change line style based on class index
        plt.plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})', linestyle=linestyle)
    
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    plt.savefig(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_roc_curve.png'))
#     plt.show()
    plt.close()
    
import csv  # Import the csv module

def calculate_and_save_auc_scores_csv(test_labels_multilabel, y_pred, class_labels, result_out, model_name, layer_name, epoch):
    """
    Calculate AUC scores for each class and save them to a CSV file.
    
    Args:
        test_labels_multilabel (array-like): True labels in categorical format.
        y_pred (array-like): Predicted labels.
        class_labels (array-like): List of class labels.
        result_out (str): Path to save the AUC scores CSV file.
    """
    auc_scores = roc_auc_score(test_labels_multilabel, y_pred, average=None)
    auc_scores_dict = {class_labels[i]: auc_scores[i] for i in range(len(class_labels))}
    
    # Print the AUC scores
    print("AUC Scores:")
    for label, score in auc_scores_dict.items():
        print(f"{label}: {score}")
    
    # Save the AUC scores to a CSV file
    auc_scores_file = os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_auc_scores.csv')
    with open(auc_scores_file, 'w', newline='') as f:  # Use newline='' for better compatibility
        writer = csv.writer(f)
        writer.writerow(['Class', 'AUC Score'])
        for label, score in auc_scores_dict.items():
            writer.writerow([label, score])

    
def calculate_and_save_auc_scores(test_labels_multilabel, y_pred, class_labels, result_out,model_name, layer_name, epoch):
    """
    Calculate AUC scores for each class and save them to a file.
    
    Args:
        test_labels_multilabel (array-like): True labels in categorical format.
        y_pred (array-like): Predicted labels.
        class_labels (array-like): List of class labels.
        result_out (str): Path to save the AUC scores file.
    """
    auc_scores = roc_auc_score(test_labels_multilabel, y_pred, average=None)
    auc_scores_dict = {class_labels[i]: auc_scores[i] for i in range(len(class_labels))}
    
#     # Print the AUC scores
#     print("AUC Scores:")
#     for label, score in auc_scores_dict.items():
#         print(f"{label}: {score}")
    
    # Save the AUC scores to a file
    auc_scores_file = os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_auc_scores.txt')
    with open(auc_scores_file, 'w') as f:
        f.write("AUC Scores:\n")
        for label, score in auc_scores_dict.items():
            f.write(f"{label}: {score}\n")


def plot_confusion_matrix(y_true, y_pred, class_labels, result_out,model_name, layer_name, epoch):
    """
    Plot and save a confusion matrix.
    confusion_matrix_stacked = confusion_matrix(test_labels_encoded, y_pred_multiclass)
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        class_labels (array-like): List of class labels.
        result_out (str): Path to save the plot.
    """
    confusion_matrix_multiclass = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_multiclass, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.title('Confusion Matrix')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0)
    
    plt.savefig(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_confusion_matrix.png'))
#     plt.show()
    plt.close()
    
# # ==================================================================
def classificatioReport(model, X_test, test_labels_encoded, result_out,model_name, layer_name, epoch):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert y_pred_stacked to multiclass format
    y_pred_multiclass = np.argmax(y_pred, axis=1)

    # Generate a classification report
    report = classification_report(test_labels_encoded, y_pred_multiclass)
        
    # Calculate accuracy
    accuracy = accuracy_score(test_labels_encoded, y_pred_multiclass)

    # Calculate precision, recall, and f1-score
    precision = precision_score(test_labels_encoded, y_pred_multiclass, average='weighted')
    recall = recall_score(test_labels_encoded, y_pred_multiclass, average='weighted')
    f1 = f1_score(test_labels_encoded, y_pred_multiclass, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_labels_encoded, y_pred_multiclass)

    # Calculate sensitivity and specificity for each class
    num_classes = conf_matrix.shape[0]
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - tp - fn - fp

        sensitivity[i] = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity[i] = tn / (tn + fp) if tn + fp > 0 else 0.0

    with open(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_classification_report.txt'), 'w') as f:
        f.write("Model Metrics: {}\n".format(model_name))
        f.write("Accuracy: {}\n".format(accuracy))
        f.write("Precision: {}\n".format(precision))
        f.write("Recall: {}\n".format(recall))
        f.write("F1-Score: {}\n".format(f1))
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf_matrix, separator=', '))
        f.write("\nSensitivity for each class:\n")
        f.write(np.array2string(sensitivity, separator=', '))
        f.write("\nSpecificity for each class:\n")
        f.write(np.array2string(specificity, separator=', '))
        f.write(f"\n{model_name} Classification Report\n")
        f.write(report)

def classificatioReport_csv(model, X_test, test_labels_encoded, result_out,model_name, layer_name, epoch):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert y_pred_stacked to multiclass format
    y_pred_multiclass = np.argmax(y_pred, axis=1)

    # Generate a classification report
    report = classification_report(test_labels_encoded, y_pred_multiclass)
        
    # Calculate accuracy
    accuracy = accuracy_score(test_labels_encoded, y_pred_multiclass)

    # Calculate precision, recall, and f1-score
    precision = precision_score(test_labels_encoded, y_pred_multiclass, average='weighted')
    recall = recall_score(test_labels_encoded, y_pred_multiclass, average='weighted')
    f1 = f1_score(test_labels_encoded, y_pred_multiclass, average='weighted')

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_labels_encoded, y_pred_multiclass)

    # Calculate sensitivity and specificity for each class
    num_classes = conf_matrix.shape[0]
    sensitivity = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fn = np.sum(conf_matrix[i, :]) - tp
        fp = np.sum(conf_matrix[:, i]) - tp
        tn = np.sum(conf_matrix) - tp - fn - fp

        sensitivity[i] = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity[i] = tn / (tn + fp) if tn + fp > 0 else 0.0
        
    # Convert classification report to a DataFrame
    report_df = pd.DataFrame(classification_report(test_labels_encoded, y_pred_multiclass, output_dict=True)).transpose()

    # Save classification report to CSV
    report_df.to_csv(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_classification_report.csv'), index=True)

    # Create a DataFrame for metrics without specifying index
    metrics_dict = {
        'Model Metrics': str(model_name),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    }
    metrics_df = pd.DataFrame(metrics_dict)

    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_metrics.csv'), index=False)


        
def accuracy_loss_plot(result_out, epoch, history, model_name, layer_name):
    plt.plot(history.history['accuracy'], linestyle='--')
    plt.plot(history.history['val_accuracy'], linestyle=':')
#     plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_accuracy_plot.png'))
#     plt.show()
    plt.close()

    plt.plot(history.history['loss'], linestyle='--')
    plt.plot(history.history['val_loss'], linestyle=':')
#     plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, f'{model_name}_{layer_name}_epoch_{epoch}_loss_plot.png'))
#     plt.show()
    plt.close()  
        
        
# Define the categories
categories = ['BCC', 'MM', 'SCC']

# Function to load and resize images from a folder
def load_and_resize_images(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith('.npy'):
            img_path = os.path.join(folder, filename)
            feature = np.load(img_path)
            label = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            
            # Check if the label is in the categories list
            if label in categories:
                images.append(feature)
                labels.append(label)
    return np.array(images), np.array(labels)


# Function to apply SMOTE oversampling to the data
def apply_smote(X, y):
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Function to normalize the data using Gaussian normalization
def normalize_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    return X_train_normalized, X_test_normalized


# ====================================================================
import os
directory_work = os.getcwd()
directory_feature = os.path.join(directory_work, 'Hien_Data_Feature_1')
model_name = 'thu_nghiem_1_BlockNet_Feature_VGG16_FC2'
result_folder = os.path.join(directory_work, model_name)
os.makedirs(result_folder, exist_ok=True)


# Define the categories
categories = ['BCC', 'MM', 'SCC']
num_categories = len(categories)
# ====================================================================


layers = ['block1_conv1', 'block1_conv2', 
          'block2_conv1', 'block2_conv2', 
          'block3_conv1', 'block3_conv2', 'block3_conv3',
          'block4_conv1', 'block4_conv2', 'block4_conv3', 
          'block5_conv1', 'block5_conv2', 'block5_conv3']

# layers = ['block1_conv1']
# layers = ['block1_conv1', 'block1_conv2']

test_metrics_block_data = []  # To store test loss and test accuracy for all epoch values

for layer_name in layers:
    print(layer_name)
    layer_result_out = os.path.join(result_folder, layer_name)
    os.makedirs(layer_result_out, exist_ok=True)
    train_images = []
    train_labels = []
    test_images = []
    test_labels = []
    for category in categories: 
        # larger dataset
        train_folder = os.path.join(directory_feature, 'BlockNetFeature_train', 
                                    'layer_fc2_vector', category, layer_name)
        test_folder = os.path.join(directory_feature, 'BlockNetFeature_test', 
                                    'layer_fc2_vector', category, layer_name)
        
        
        # Load and resize train set
        layer_image, layer_label = load_and_resize_images(train_folder)
        train_images.append(layer_image)
        train_labels.append(layer_label)
        
        # Load and resize test set
        layer_image, layer_label = load_and_resize_images(test_folder)
        test_images.append(layer_image)
        test_labels.append(layer_label)
    
    # Convert the lists to NumPy arrays
    train_images = np.concatenate(train_images)
    test_images = np.concatenate(test_images)
    
    # Reshape feature data to 1D vectors
    train_images = train_images.reshape(train_images.shape[0], -1)
    test_images = test_images.reshape(test_images.shape[0], -1)
    
    # Flatten and convert labels to strings
    train_labels = np.array([label for sublist in train_labels for label in sublist], dtype=str)
    test_labels = np.array([label for sublist in test_labels for label in sublist], dtype=str)

    
    # Apply SMOTE oversampling to the train set
    # train_images, train_labels = apply_smote(train_images, train_labels)
    
    # Normalize the feature data using Gaussian normalization
    train_images_normalized, test_images_normalized = normalize_data(train_images, test_images)
    
    # Split training data into training, validation, and testing sets
    test_images_normalized, val_images_normalized, test_labels, val_labels = train_test_split(test_images_normalized, test_labels, test_size=0.5, random_state=42)


    # Convert labels to categorical
    label_encoder = LabelEncoder()
    train_labels_encoded = label_encoder.fit_transform(train_labels)
    test_labels_encoded = label_encoder.transform(test_labels)
    val_labels_encoded = label_encoder.transform(val_labels)

    num_categories = len(label_encoder.classes_)  # Get the number of unique classes

    train_labels_categorical = to_categorical(train_labels_encoded, num_classes=num_categories)
    test_labels_categorical = to_categorical(test_labels_encoded, num_classes=num_categories)
    val_labels_categorical = to_categorical(val_labels_encoded, num_classes=num_categories)


    # Inside your loop for different epoch values
    test_metrics_data = []  # To store test loss and test accuracy for all epoch values
    train_metrics_data = []
    # Define a list of epoch values to try
    # epoch_values = [5, 10, 15, 25, 40, 65, 105]
    
    # Define a list of batch sizes to try
    batch_sizes = [8, 16, 32, 64, 128, 256]
    # batch_sizes = [8, 16]


    for batch_size in batch_sizes:
        # Create and compile your model
        batch_size_result_out = os.path.join(layer_result_out, 'batch_size_' + str(batch_size))
        os.makedirs(batch_size_result_out, exist_ok=True)
        
        test_metrics_data = []  # To store test loss and test accuracy for all epoch values
        performance_metrics_data = []
    
    
    
        epoch_values = [2, 4, 8, 10, 14, 20, 40, 60, 80, 100]
        # epoch_values = [2, 3]
    
        # Inside your loop for different epoch values
        for epoch in epoch_values:
            result_out = os.path.join(batch_size_result_out, 'epoch_' + str(epoch))
            os.makedirs(result_out, exist_ok=True)

            # model_0 = create_keras_model_1(train_images_normalized, train_labels)
            model_1 = create_keras_model_2(train_images_normalized, train_labels)

            # Record start time
            start_time = datetime.now()

            # Create a callback to record loss and accuracy during training for both validation and test sets
            callback = tf.keras.callbacks.CSVLogger(os.path.join(result_out, f'model_1_{layer_name}_epoch_{epoch}_training_log.csv'))
            
            # Record the start time of training
            history_1 = model_1.fit(train_images_normalized, train_labels_categorical, epochs=epoch, batch_size= batch_size,
                                    validation_data=(val_images_normalized, val_labels_categorical), verbose=2,
                                    callbacks=[callback])  # Use the callback here

            # Record end time
            end_time = datetime.now()
            epoch_training_time = (end_time - start_time).total_seconds()

            # Save model_1
#             model_1.save(os.path.join(result_out, f'model_1_{layer_name}_epoch_{epoch}.h5'))
            model_1.save(os.path.join(result_out, f'model_1_{layer_name}_epoch_{epoch}.keras'))

            
            # Make predictions on the test set
            y_pred = model_1.predict(test_images_normalized)
            
            # Convert test labels to binary format for AUC calculation
            test_labels_binary = label_binarize(test_labels_encoded, classes=range(num_categories))

            # Calculate AUC scores for each class
            auc_scores = roc_auc_score(test_labels_binary, y_pred, average=None)
        
            # Convert y_pred to multiclass format
            y_pred_multiclass = np.argmax(y_pred, axis=1)
            
            # Calculate accuracy, precision, recall, and f1-score
            accuracy = accuracy_score(test_labels_encoded, y_pred_multiclass)
            precision = precision_score(test_labels_encoded, y_pred_multiclass, average='weighted')
            recall = recall_score(test_labels_encoded, y_pred_multiclass, average='weighted')
            f1 = f1_score(test_labels_encoded, y_pred_multiclass, average='weighted')
        
        
            conf_matrix = confusion_matrix(test_labels_encoded, y_pred_multiclass)

            # Calculate sensitivity and specificity
            sensitivity, specificity = cal_sensitivity_specificity(conf_matrix, num_categories)
        
            # Access loss and accuracy from history for both validation and test sets
            loss = history_1.history['loss']
            val_loss = history_1.history['val_loss']
            accuracy = history_1.history['accuracy']
            val_accuracy = history_1.history['val_accuracy']

            # Append the train metrics for this epoch to the list
            train_metrics_data.append({'layer_name': layer_name, 
                                       'batch_size': batch_size,
                                       'Epoch': epoch, 
                                       'Train Loss': loss, 
                                       'Train Accuracy': accuracy, 
                                        'Val Loss': val_loss, 
                                        'Val Accuracy': val_accuracy})
            
            # Evaluate the model on the test set to get test loss and accuracy
            test_loss, test_accuracy = model_1.evaluate(test_images_normalized, test_labels_categorical, verbose=0)
            
            # Append the test metrics for this epoch to the list
            test_metrics_data.append({'layer_name': layer_name, 
                                      'batch_size': batch_size,
                                      'Epoch': epoch, 
                                      'Test Loss': test_loss, 
                                      'Test Accuracy': test_accuracy})

            test_metrics_block_data.append({'layer_name': layer_name, 
                                            'batch_size': batch_size,
                                            'Epoch': epoch, 
                                            'Test Loss': test_loss, 
                                            'Test Accuracy': test_accuracy,
                                            'Accuracy': accuracy,
                                            'precision': precision,
                                            'recall': recall,
                                            'f1_score': f1,
                                            'sensitivity': sensitivity,
                                            'specificity':specificity,
                                            'auc_scores': auc_scores,
                                            'epoch_training_time': epoch_training_time
                                            })
            
            # Now you can save or analyze these loss and accuracy values as needed
            classificatioReport(model_1, test_images_normalized, test_labels_encoded, result_out, 'model_1', layer_name, epoch)
            classificatioReport_csv(model_1, test_images_normalized, test_labels_encoded, result_out, 'model_1', layer_name, epoch)
            
            # Plot the training history for each model
            accuracy_loss_plot(result_out, epoch, history_1, 'model_1', layer_name)
            
            class_labels = label_encoder.classes_
            test_labels_multilabel = label_binarize(test_labels_encoded, classes=np.arange(num_categories))
            
            # Generate the confusion matrix for the model_1
            y_pred = model_1.predict(test_images_normalized)
            
            # Convert y_pred to multiclass format
            y_pred_multiclass = np.argmax(y_pred, axis=1)        
            plot_confusion_matrix(test_labels_encoded, y_pred_multiclass, class_labels, result_out, 'model_1', layer_name, epoch)
            
            # Calculate AUC scores for each class and save them to a file 
            calculate_and_save_auc_scores(test_labels_multilabel, y_pred, class_labels, result_out, 'model_1', layer_name, epoch)
            
            # Calculate AUC scores for each class and save them to a csv file 
            calculate_and_save_auc_scores_csv(test_labels_multilabel, y_pred, class_labels, result_out, 'model_1', layer_name, epoch)
            
            
            plot_roc_curve_multilabel(test_labels_multilabel, y_pred, class_labels, result_out, 'model_1', layer_name, epoch)
            
            # Save Sensitivity and Specificity to a CSV file
            save_sensitivity_specificity(model_1, test_images_normalized, test_labels_encoded, result_out, 'model_1', layer_name, epoch)
        
        
        # Create a DataFrame from the list of train metrics data
        train_metrics_df = pd.DataFrame(train_metrics_data)
        # Define the path for the CSV file (outside the epoch loop)
        train_metrics_csv = os.path.join(layer_result_out, f'{layer_name}_train_metrics.csv')
        # Save the DataFrame to a CSV file
        train_metrics_df.to_csv(train_metrics_csv, index=False)
        
        # Create a DataFrame from the list of test metrics data
        test_metrics_df = pd.DataFrame(test_metrics_data)
        # Define the path for the CSV file (outside the epoch loop)
        test_metrics_csv = os.path.join(layer_result_out, f'{layer_name}_test_metrics.csv')
        # Save the DataFrame to a CSV file
        test_metrics_df.to_csv(test_metrics_csv, index=False)


# Create a DataFrame from the list of test metrics data
test_block_metrics_df = pd.DataFrame(test_metrics_block_data)
# Define the path for the CSV file (outside the epoch loop)
test_block_metrics_csv = os.path.join(result_folder, 'test_block_metrics.csv')
# Save the DataFrame to a CSV file
test_block_metrics_df.to_csv(test_block_metrics_csv, index=False)
    