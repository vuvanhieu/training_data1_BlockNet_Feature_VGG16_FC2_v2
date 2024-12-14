# method_10_FAIR2023_SkinLesion_FC2_Compare_ResNet101_FC2
# test_Hien_VGG19_FC_2_full
# train_Hien_VGG19_FC_2_full

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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# # ==================================================================

import os
import pandas as pd


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


def calculate_sensitivity_specificity(confusion_matrix_metrics, class_labels, result_out, epoch, model_name):
    num_classes = len(class_labels)
    sensitivity = []
    specificity = []

    for i in range(num_classes):
        tp = confusion_matrix_metrics[i, i]
        fn = sum(confusion_matrix_metrics[i, :]) - tp
        fp = sum(confusion_matrix_metrics[:, i]) - tp
        tn = sum(sum(confusion_matrix_metrics)) - tp - fn - fp

        sensitivity_i = tp / (tp + fn) if tp + fn > 0 else 0.0
        specificity_i = tn / (tn + fp) if tn + fp > 0 else 0.0

        sensitivity.append(sensitivity_i)
        specificity.append(specificity_i)

    results = pd.DataFrame({
        'Class': class_labels,
        'Sensitivity': sensitivity,
        'Specificity': specificity
    })

    results_file = os.path.join(result_out, f'epoch_{epoch}_{model_name}_sensitivity_specificity.csv')
    results.to_csv(results_file, index=False)

    return results


def plot_and_save_confusion_matrix(result_out, class_labels, confusion_matrix_metrics, epoch, model_name):
    # Plot the confusion matrix with class labels
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_metrics, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.title('Confusion Matrix - Model')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0)
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}_confusion_matrix.png'))
#     plt.show()
    plt.close()


def plot_roc_curve(result_out, class_labels, test_labels_multilabel, epoch, model_name):
    # Calculate the false positive rate (FPR) and true positive rate (TPR) for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(len(class_labels)):
        fpr[i], tpr[i], _ = roc_curve(test_labels_multilabel[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
        
    plt.figure(figsize=(8, 6))
    for i in range(len(class_labels)):
        linestyle = '--' if i % 2 == 0 else '-'
        plt.plot(fpr[i], tpr[i], label=f'{class_labels[i]} (AUC = {roc_auc[i]:.2f})', linestyle=linestyle)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve - Stacked Model')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}_roc_curve.png'))
#     plt.show()
    plt.close()

def save_auc_scores_to_csv(result_out, class_labels, test_labels_multilabel, y_pred, epoch, model_name):
    # Calculate the AUC for each class
    auc_scores = roc_auc_score(test_labels_multilabel, y_pred, average=None)

    # Create a dictionary to map class labels to AUC scores
    auc_scores_dict = {class_labels[i]: auc_scores[i] for i in range(len(class_labels))}
    
    auc_scores_file = os.path.join(result_out, f'epoch_{epoch}_{model_name}_auc_scores.csv')
    auc_df = pd.DataFrame({'Class': class_labels, 'AUC Score': list(auc_scores_dict.values())})
    auc_df.to_csv(auc_scores_file, index=False)


def classificatioReport(model, X_test, test_labels_encoded, result_out, index, epoch, model_name):
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

    with open(os.path.join(result_out, f'epoch_{epoch}_{model_name}_classification_report.txt'.format(index)), 'w') as f:
        f.write("Model Metrics: {}\n".format(index))
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
        f.write("\nModel Classification Report:{}\n".format(index))
        f.write(report)



def classificatioReportToCSV(model, X_test, test_labels_encoded, result_out, index, epoch, model_name):
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Convert y_pred to multiclass format
    y_pred_multiclass = np.argmax(y_pred, axis=1)

    # Generate a classification report
    report = classification_report(test_labels_encoded, y_pred_multiclass, output_dict=True)
        

    # Create a DataFrame to store the classification report
    report_df = pd.DataFrame(report).T
    report_df.index.name = 'Class'
    report_df.reset_index(inplace=True)

    # Save the classification report to a CSV file
    report_file = os.path.join(result_out, f'epoch_{epoch}_{model_name}_classification_report.csv'.format(index))
    report_df.to_csv(report_file, index=False)

    
    
def accuracy_loss_plot(result_out, history, epoch, model_name):
    plt.plot(history.history['accuracy'], linestyle='--')
    plt.plot(history.history['val_accuracy'], linestyle=':')
#     plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}_accuracy_plot.png'))
    # plt.show()
    plt.close()

    plt.plot(history.history['loss'], linestyle='--')
    plt.plot(history.history['val_loss'], linestyle=':')
#     plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}_loss_plot.png'))
#     plt.show()
    plt.close()  
    

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



def load_and_resize_images(folder, labels):
    X = []
    y = []
    class_folders = sorted(os.listdir(folder))
    for class_folder in class_folders:
        if class_folder in labels:
            class_path = os.path.join(folder, class_folder)
            if os.path.isdir(class_path):
                for filename in os.listdir(class_path):
                    if filename.endswith('.npy'):
                        img_path = os.path.join(class_path, filename)
                        feature = np.load(img_path)
                        X.append(feature)
                        y.append(class_folder)
    return np.array(X), np.array(y)


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
model_name = 'thu_nghiem_1_classification_CNN_Feature_ResNet152_FC2'
result_folder = os.path.join(directory_work, model_name)
os.makedirs(result_folder, exist_ok=True)

# Define the categories
categories = ['BCC', 'MM', 'SCC']
num_categories = len(categories)

# ====================================================================    
train_folder = os.path.join(directory_feature, 'Feature_ResNet152_FC2_train')
test_folder = os.path.join(directory_feature, 'Feature_ResNet152_FC2_test')
# Load and resize train set
# train_images, train_labels = load_and_resize_images(train_folder)
train_images, train_labels = load_and_resize_images(train_folder, categories)

# Load and resize test set
# test_images, test_labels = load_and_resize_images(test_folder)
# Load and resize test set
test_images, test_labels = load_and_resize_images(test_folder, categories)
    
# Reshape feature data to 1D vectors
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)

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


# Define a list of epoch values to try
# epoch_values = [5, 10, 15, 25, 40, 65, 105]
epoch_values = [2, 4, 8, 10, 14, 20, 40, 60, 80, 100]
# epoch_values = [2, 4]


# Define a list of batch sizes to try
batch_sizes = [8, 16, 32, 64, 128, 256]
# batch_sizes = [8, 16]

test_metrics_block_data=[]

for batch_size in batch_sizes:
    # Create and compile your model
    batch_size_result_out = os.path.join(result_folder, 'batch_size_' + str(batch_size))
    os.makedirs(batch_size_result_out, exist_ok=True)
    
    test_metrics_data = []  # To store test loss and test accuracy for all epoch values
    performance_metrics_data = []
    
    for epoch in epoch_values:
        result_out = os.path.join(batch_size_result_out, 'epoch_'+ str(epoch))
        os.makedirs(result_out, exist_ok=True)

        index = 0
        model_1 = create_keras_model_2(train_images_normalized, train_labels)

        # Record the start time of training
        start_time = time.time()
        
        history_1 = model_1.fit(train_images_normalized, train_labels_categorical, epochs=epoch, batch_size= batch_size, 
                                    validation_data=(val_images_normalized, val_labels_categorical), verbose=2)
        
        # Record the end time of training
        end_time = time.time()
        
        model_1.save(os.path.join(result_out, f'epoch_{epoch}_{model_name}_model_1.h5'))  # Save model_2

        # Calculate the training time
        epoch_training_time = end_time - start_time
        
        # After the epoch is completed, get the test loss and test accuracy
        test_loss, test_accuracy = model_1.evaluate(test_images_normalized, test_labels_categorical, verbose=0)
        
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


        # Append the test metrics for this epoch to the list
        test_metrics_data.append({'Model Name': model_name,
                                  'batch_sizes': batch_size,
                                  'Epoch': epoch, 
                                  'Test Loss': test_loss, 
                                  'Test Accuracy': test_accuracy,
                                  'epoch_training_time': epoch_training_time})
        
        
        performance_metrics_data.append({'Model Name': model_name,
                                         'batch_sizes': batch_size,
                                         'Epoch': epoch, 
                                         'accuracy': accuracy, 
                                         'precision': precision,
                                         'recall': recall, 
                                         'f1':f1})
        
        # Append to test_metrics_block_data
        test_metrics_block_data.append({'Model Name': model_name,
                                         'batch_sizes': batch_size,
                                         'Epoch': epoch, 
                                         'accuracy': accuracy, 
                                         'precision': precision,
                                         'recall': recall, 
                                         'f1':f1,
                                         'Test Loss': test_loss, 
                                         'Test Accuracy': test_accuracy,
                                         'Sensitivity': sensitivity,
                                         'Specificity': specificity,
                                         'auc_scores': auc_scores,
                                         'epoch_training_time': epoch_training_time})
        
        
        
        
        classificatioReport(model_1, test_images_normalized, test_labels_encoded, result_out, index, epoch, model_name)
        
        classificatioReportToCSV(model_1, test_images_normalized, test_labels_encoded, result_out, index, epoch, model_name)
        # Plot the training history for each model
        #==================================================
        accuracy_loss_plot(result_out, history_1, epoch, model_name)
        #==================================================
        # Make predictions on the test set
        y_pred = model_1.predict(test_images_normalized)

        # Convert y_pred to multiclass format
        y_pred_multiclass = np.argmax(y_pred, axis=1)
            
        # Generate the confusion matrix for the stacked model
        confusion_matrix_metrics = confusion_matrix(test_labels_encoded, y_pred_multiclass)
        # Get the class labels
        class_labels = label_encoder.classes_

        # Plot and save the confusion matrix
        plot_and_save_confusion_matrix(result_out, class_labels, confusion_matrix_metrics, epoch, model_name)
        
        test_labels_multilabel = to_categorical(test_labels_encoded, num_classes=num_categories)

        # Add the following function for plotting ROC curves 
        plot_roc_curve(result_out, class_labels, test_labels_multilabel, epoch, model_name)
        
        # Add the following function for saving AUC scores to a CSV file in your result_out directory
        save_auc_scores_to_csv(result_out, class_labels, test_labels_multilabel, y_pred, epoch, model_name)
        
        # Calculate sensitivity and specificity for each class and save to CSV
        sensitivity_specificity_results = calculate_sensitivity_specificity(confusion_matrix_metrics, class_labels, result_out, epoch, model_name)

    # # After all epochs, save epoch metrics to a CSV file
    test_metrics_df = pd.DataFrame(test_metrics_data)  
    # Define the path for the CSV file (outside the epoch loop)
    test_metrics_csv = os.path.join(result_folder, f'{model_name}_epoch_metrics.csv')
    # Save the DataFrame to a CSV file
    test_metrics_df.to_csv(test_metrics_csv, index=False)

    # Save performance metrics to a CSV file
    # Create a DataFrame from the list of test performance data
    performance_metrics_df = pd.DataFrame(performance_metrics_data)
    # Define the path for the CSV file (outside the epoch loop)
    performance_metrics_csv = os.path.join(result_folder, f'{model_name}_epoch_performance_metrics.csv')
    # Save the DataFrame to a CSV file
    performance_metrics_df.to_csv(performance_metrics_csv, index=False)
    

# Create a DataFrame from the list of test metrics block data
test_block_metrics_df = pd.DataFrame(test_metrics_block_data)
# Define the path for the CSV file (outside the epoch loop)
test_block_metrics_csv = os.path.join(result_folder, f'{model_name}_test_batch_size_metrics.csv')
# Save the DataFrame to a CSV file
test_block_metrics_df.to_csv(test_block_metrics_csv, index=False)