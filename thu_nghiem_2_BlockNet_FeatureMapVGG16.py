import os
import numpy as np
import pandas as pd
import time
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, roc_curve, auc, roc_auc_score
from tensorflow.keras.applications.vgg16 import VGG16
from sklearn.preprocessing import LabelEncoder, label_binarize


# Set the GPU device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# ==================================================================

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


# Define functions to calculate sensitivity and specificity
def calculate_sensitivity_specificity(confusion_matrix_metrics, class_labels, result_out, epoch, model_name):
    num_classes = len(class_labels)
    sensitivity = []
    specificity = []

    for i in range(num_classes):
        tp = confusion_matrix_metrics[i, i]
        fn = np.sum(confusion_matrix_metrics[i, :]) - tp
        fp = np.sum(confusion_matrix_metrics[:, i]) - tp
        tn = np.sum(confusion_matrix_metrics) - tp - fn - fp

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

def plot_and_save_confusion_matrix(result_out, class_labels, confusion_matrix_metrics, epoch, model_name):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix_metrics, annot=True, fmt='d', cmap='Blues', cbar=False)
#     plt.title('Confusion Matrix - Model')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
    plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0)
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}_confusion_matrix.png'))
    plt.close()

def plot_roc_curve(result_out, class_labels, test_labels_multilabel, epoch, model_name):
    plt.figure(figsize=(8, 6))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_classes = len(class_labels)

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(test_labels_multilabel[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label=f'{class_labels[i]} (AUC = {roc_auc[i]:0.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
#     plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}_roc_curve.png'))
    plt.close()

def save_auc_scores_to_csv(result_out, class_labels, test_labels_multilabel, y_pred, epoch, model_name):
    num_classes = len(class_labels)
    auc_scores = []

    for i in range(num_classes):
        auc_i = roc_auc_score(test_labels_multilabel[:, i], y_pred[:, i])
        auc_scores.append(auc_i)

    results = pd.DataFrame({
        'Class': class_labels,
        'AUC Score': auc_scores
    })

    results_file = os.path.join(result_out, f'epoch_{epoch}_{model_name}_auc_scores.csv')
    results.to_csv(results_file, index=False)

def classificationReport(model, X_test, test_labels_encoded, result_out, index, epoch, model_name):
    y_pred = model.predict(X_test)
    y_pred_multiclass = np.argmax(y_pred, axis=1)
    class_labels = list(label_encoder.classes_)

    accuracy = accuracy_score(test_labels_encoded, y_pred_multiclass)
    precision = precision_score(test_labels_encoded, y_pred_multiclass, average='weighted')
    recall = recall_score(test_labels_encoded, y_pred_multiclass, average='weighted')
    f1 = f1_score(test_labels_encoded, y_pred_multiclass, average='weighted')

    classification_report_dict = classification_report(test_labels_encoded, y_pred_multiclass, target_names=class_labels, output_dict=True)

    classification_report_df = pd.DataFrame(classification_report_dict)
    classification_report_df.to_csv(os.path.join(result_out, f'epoch_{epoch}_{model_name}_classification_report_{index}.csv'))

    results = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

    results_file = os.path.join(result_out, f'epoch_{epoch}_{model_name}_classification_results_{index}.csv')

    with open(results_file, 'w') as file:
        for key, value in results.items():
            file.write(f'{key}: {value}\n')

def classificationReportToCSV(model, X_test, test_labels_encoded, result_out, index, epoch, model_name):
    y_pred = model.predict(X_test)
    y_pred_multiclass = np.argmax(y_pred, axis=1)
    class_labels = list(label_encoder.classes_)

    classification_report_dict = classification_report(test_labels_encoded, y_pred_multiclass, target_names=class_labels, output_dict=True)

    classification_report_df = pd.DataFrame(classification_report_dict)
    classification_report_df.to_csv(os.path.join(result_out, f'epoch_{epoch}_{model_name}_classification_report_{index}.csv'))

def accuracy_loss_plot(result_out, history, epoch, model_name):    
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
#     plt.title('Model accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('accuracy_plot.png')  # Save the plot as an image
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}_accuracy_plot.png'))
    plt.close()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
#     plt.title('Model loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss_plot.png')  # Save the plot as an image
    plt.savefig(os.path.join(result_out, f'epoch_{epoch}_{model_name}loss_plot.png'))
    plt.close()
    
# ==================================================================


# Define the directory paths and model name
directory_work = os.getcwd()
directory_feature = os.path.join(directory_work, 'Hien_Data_Feature_1')
model_name = 'thu_nghiem_2_BlockNet_FeatureMapVGG16'
result_folder = os.path.join(directory_work, model_name)
os.makedirs(result_folder, exist_ok=True)

# Define the categories
categories = ['BCC', 'MM', 'SCC']

num_categories = len(categories)

num_classes = len(categories)

# label_encoder = LabelEncoder()

# class_labels = list(label_encoder.classes_)


def load_and_resize_images(folder, target_size=(224, 224), image_extensions=('.jpg', '.jpeg', '.png')):
    images = []
    labels = []
    for category in categories:
        category_folder = os.path.join(folder)
        for filename in os.listdir(category_folder):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                img_path = os.path.join(category_folder, filename)
                label = category
                img = load_img(img_path, target_size=target_size)  # Load and resize the image
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(label)
    return images, labels




train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Specify ImageNet mean values for centering
train_datagen.mean = [123.68, 116.779, 103.939]
test_datagen.mean = [123.68, 116.779, 103.939]
val_datagen.mean = [123.68, 116.779, 103.939]


# Load the pre-trained VGG16 model
base_model = VGG16(weights='imagenet', include_top=False)

# Add custom classification layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

layers = ['block1_conv1', 'block1_conv2', 
          'block2_conv1', 'block2_conv2', 
          'block3_conv1', 'block3_conv2', 'block3_conv3',
          'block4_conv1', 'block4_conv2', 'block4_conv3', 
          'block5_conv1', 'block5_conv2', 'block5_conv3']

layers = ['block1_conv1', 'block1_conv2', 
          'block2_conv1', 'block2_conv2']

# Define a list of epoch values to try
# epoch_values = [2, 4, 8, 10, 12, 14, 16, 18, 20, 22, 24, 28, 28, 30,32, 34, 36, 38, 40]
epoch_values = [2, 4, 8, 10, 14, 20, 40, 60, 80, 100]

# epoch_values = [2]

# Define a list of batch sizes to try
# batch_sizes = [8, 16, 32, 64, 128, 256]

batch_sizes = [8]

test_metrics_data = []  # To store test loss and test accuracy for all epoch values
performance_metrics_data = []
test_metrics_block_data=[]

for batch_size in batch_sizes:
    # Create and compile your model
    batch_size_result_out = os.path.join(result_folder, 'batch_size_' + str(batch_size))
    os.makedirs(batch_size_result_out, exist_ok=True)
    
    for layer_name in layers:
        print(layer_name)
        layer_result_out = os.path.join(batch_size_result_out, layer_name)
        os.makedirs(layer_result_out, exist_ok=True)
        train_images = []
        train_labels = []
        test_images = []
        test_labels = []
        for category in categories: 
            # larger dataset
            train_folder = os.path.join(directory_feature, 'BlockNetFeature_train', 'layer_feature_map',
                                        category, layer_name)
            test_folder = os.path.join(directory_feature, 'BlockNetFeature_test', 'layer_feature_map',
                                        category, layer_name)
    
    
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

        
        # Split the test data into validation and test sets
        val_image, test_images, val_labels, test_labels = train_test_split(test_images, 
                                                                           test_labels, 
                                                                           test_size=0.5, random_state=42)

        # Reshape train_images and test_images
        train_images = np.array(train_images).reshape(-1, 224, 224, 3)
        test_images = np.array(test_images).reshape(-1, 224, 224, 3)
        val_images = np.array(val_image).reshape(-1, 224, 224, 3)

        # Convert string labels to numerical labels
        label_encoder = LabelEncoder()
        train_labels_encoded = label_encoder.fit_transform(train_labels)
        test_labels_encoded = label_encoder.transform(test_labels)
        val_labels_encoded = label_encoder.transform(val_labels)
        class_labels = list(label_encoder.classes_)


        # Convert train_labels and test_labels to one-hot encoded format

        train_labels_categorical = to_categorical(train_labels_encoded, num_classes)
        test_labels_categorical = to_categorical(test_labels_encoded, num_classes)
        val_labels_categorical = to_categorical(val_labels_encoded, num_classes)

        for epoch in epoch_values:

            result_out = os.path.join(layer_result_out, 'epoch_' + str(epoch))
            os.makedirs(result_out, exist_ok=True)

            # Create data generators with the current batch size
            train_datagen = ImageDataGenerator(rescale=1. / 255)
            val_datagen = ImageDataGenerator(rescale=1. / 255)
            test_datagen = ImageDataGenerator(rescale=1. / 255)

            train_generator = train_datagen.flow(train_images, train_labels_encoded, 
                                                 batch_size=batch_size)
            val_generator = val_datagen.flow(val_images, val_labels_encoded, 
                                             batch_size=batch_size)
            test_generator = test_datagen.flow(test_images, test_labels_encoded, 
                                               batch_size=batch_size)

            index = 0

            # Record the start time of training
            start_time = time.time()

            # Train the model
            history = model.fit(train_generator, epochs=epoch, validation_data=val_generator, verbose=1)

            # Record the end time of training
            end_time = time.time()

            # Calculate the training time
            epoch_training_time = end_time - start_time

            # Save the model
#             model.save(os.path.join(result_folder, f'{model_name}_batch_size_{batch_size}_epoch_{epoch}_layer_{layer_name}.h5'))

            # Test the model
            test_loss, test_accuracy = model.evaluate(test_generator, verbose=0)

            # Append the test metrics for this epoch to the list
            test_metrics_data.append({'Model Name': model_name, 
                                      'batch_size': batch_size,
                                      'layer_name': layer_name,
                                      'Epoch': epoch, 
                                      'Test Loss': test_loss, 
                                      'Test Accuracy': test_accuracy, 
                                      'Training Time (s)': epoch_training_time})

            # Make predictions on the test set
            y_pred = model.predict(test_generator)
            y_pred_multiclass = np.argmax(y_pred, axis=1)

            conf_matrix = confusion_matrix(test_labels_encoded, y_pred_multiclass)
            
            # Calculate sensitivity and specificity
            sensitivity, specificity = cal_sensitivity_specificity(conf_matrix, num_categories)
        
            # Convert test labels to binary format for AUC calculation
            test_labels_binary = label_binarize(test_labels_encoded, classes=range(num_categories))

            
            # Calculate AUC scores for each class
            auc_scores = roc_auc_score(test_labels_binary, y_pred, average=None)
        
            # Calculate accuracy, precision, recall, and f1-score
            accuracy = accuracy_score(test_labels_encoded, y_pred_multiclass)
            precision = precision_score(test_labels_encoded, y_pred_multiclass, average='weighted')
            recall = recall_score(test_labels_encoded, y_pred_multiclass, average='weighted')
            f1 = f1_score(test_labels_encoded, y_pred_multiclass, average='weighted')


            test_metrics_block_data.append({'Model Name': model_name, 
                                            'batch_size': str(batch_size), 
                                            'layer_name': layer_name,
                                            'Epoch': epoch, 
                                            'Test Loss': test_loss, 
                                            'Test Accuracy': test_accuracy,
                                            'accuracy': accuracy,
                                            'precision': precision,
                                            'recall': recall,
                                            'f1': f1,
                                            'Sensitivity': sensitivity,
                                            'Specificity': specificity,
                                            'auc_scores': auc_scores,
                                            'Training Time (s)': epoch_training_time
                                            })

            # Confusion Matrix
            confusion_matrix_metrics = confusion_matrix(test_labels_encoded, y_pred_multiclass)
            plot_and_save_confusion_matrix(result_out, categories, confusion_matrix_metrics, epoch, model_name)

            # Sensitivity and Specificity
            calculate_sensitivity_specificity(confusion_matrix_metrics, categories, result_out, epoch, model_name)

            # ROC Curve
            test_labels_multilabel = to_categorical(test_labels_encoded, num_classes)
            plot_roc_curve(result_out, class_labels, test_labels_multilabel, epoch, model_name)

            calculate_sensitivity_specificity(confusion_matrix_metrics, class_labels, result_out, epoch, model_name)
            plot_and_save_confusion_matrix(result_out, class_labels, confusion_matrix_metrics, epoch, model_name)
            plot_roc_curve(result_out, class_labels, test_labels_multilabel, epoch, model_name)
            save_auc_scores_to_csv(result_out, class_labels, test_labels_multilabel, y_pred, epoch, model_name)
            classificationReport(model, test_generator, test_labels_encoded, result_out, index, epoch, model_name)
            classificationReportToCSV(model, test_generator, test_labels_encoded, result_out, index, epoch, model_name)
            accuracy_loss_plot(result_out, history, epoch, model_name)



        # After all epochs, save epoch metrics to a CSV file
        test_metrics_df = pd.DataFrame(test_metrics_data)
        test_metrics_csv = os.path.join(batch_size_result_out, f'{model_name}_epoch_metrics.csv')
        test_metrics_df.to_csv(test_metrics_csv, index=False)
    
# Create a DataFrame from the list of test metrics block data
test_block_metrics_df = pd.DataFrame(test_metrics_block_data)

# Define the path for the CSV file (outside the epoch loop)
test_block_metrics_csv = os.path.join(result_folder, f'{model_name}_test_batch_size_metrics.csv')

# Save the DataFrame to a CSV file
test_block_metrics_df.to_csv(test_block_metrics_csv, index=False)

print("done")