import os
import cv2
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from tqdm import tqdm
import joblib
from skimage.feature import hog  # HOG feature extraction
import matplotlib.pyplot as plt

# Function to load data and apply HOG feature extraction
def load_data_with_hog(directory, target_size=(128, 128), hog_params=None):
    X = []
    y = []
    labels = sorted(os.listdir(directory))
    if hog_params is None:
        hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2'
        }
    
    for label in labels:
        label_dir = os.path.join(directory, label)
        for file in tqdm(os.listdir(label_dir), desc=f"Loading {label} images"):
            image_path = os.path.join(label_dir, file)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, target_size)
                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                # Apply HOG feature extraction
                hog_features = hog(gray_image, **hog_params)
                X.append(hog_features)
                y.append(label)
    return np.array(X), np.array(y)

train_dir = 'D:/AAA/train'
validation_dir = 'D:/AAA/validation'
test_dir = 'D:/AAA/test'

# Cache file paths
train_features_file = 'train_features.npy'
train_labels_file = 'train_labels.npy'
validation_features_file = 'validation_features.npy'
validation_labels_file = 'validation_labels.npy'
test_features_file = 'test_features.npy'
test_labels_file = 'test_labels.npy'

# Load or calculate train data with HOG features
if os.path.exists(train_features_file) and os.path.exists(train_labels_file):
    X_train = np.load(train_features_file)
    y_train = np.load(train_labels_file)
else:
    X_train, y_train = load_data_with_hog(train_dir)
    np.save(train_features_file, X_train)
    np.save(train_labels_file, y_train)

# Load or calculate validation data with HOG features
if os.path.exists(validation_features_file) and os.path.exists(validation_labels_file):
    X_validation = np.load(validation_features_file)
    y_validation = np.load(validation_labels_file)
else:
    X_validation, y_validation = load_data_with_hog(validation_dir)
    np.save(validation_features_file, X_validation)
    np.save(validation_labels_file, y_validation)

# Load or calculate test data with HOG features
if os.path.exists(test_features_file) and os.path.exists(test_labels_file):
    X_test = np.load(test_features_file)
    y_test = np.load(test_labels_file)
else:
    X_test, y_test = load_data_with_hog(test_dir)
    np.save(test_features_file, X_test)
    np.save(test_labels_file, y_test)

model_file = 'random_forest_with_hog_100_10.pkl'

# Train with GridSearchCV if model file doesn't exist
if os.path.exists(model_file):
    clf = joblib.load(model_file)
else:
    clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10)
    clf.fit(X_train, y_train)
    joblib.dump(clf, model_file)


y_pred_train = clf.predict(X_train)
accuracy_train = accuracy_score(y_train, y_pred_train)

print("Training Set Accuracy:", accuracy_train)
report_validation = classification_report(y_train, y_pred_train, target_names=['Bicycle', 'Bus', 'Car', 'Motorbike', 'Truck'], zero_division=1)
print("Training Set Classification Report:")
print(report_validation)

# Evaluate the model on the validation set
y_pred_validation = clf.predict(X_validation)
accuracy_validation = accuracy_score(y_validation, y_pred_validation)

print("Validation Set Accuracy:", accuracy_validation)
report_validation = classification_report(y_validation, y_pred_validation, target_names=['Bicycle', 'Bus', 'Car', 'Motorbike', 'Truck'], zero_division=1)
print("Validation Set Classification Report:")
print(report_validation)

# Evaluate the model on the test set
y_pred_test = clf.predict(X_test)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Test Set Accuracy:", accuracy_test)
report_test = classification_report(y_test, y_pred_test, target_names=['Bicycle', 'Bus', 'Car', 'Motorbike', 'Truck'], zero_division=1)
print("Test Set Classification Report:")
print(report_test)

# Generate confusion matrix
target_names = ['Bicycle', 'Bus', 'Car', 'Motorbike', 'Truck']
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_test, normalize='true', values_format=".0%", display_labels=target_names)
plt.show()
