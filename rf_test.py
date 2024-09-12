import os
import cv2
import joblib
from skimage.feature import hog
test_dir = 'D:/AAA/test'

model_file = 'random_forest_with_hog_100_10.pkl'
clf = joblib.load(model_file)

image_path = os.path.join(test_dir, 'Motorcycle', '000023_03.jpg')
image = cv2.imread(image_path)
if image is not None:
    hog_params = {
            'orientations': 9,
            'pixels_per_cell': (8, 8),
            'cells_per_block': (2, 2),
            'block_norm': 'L2'
        }
    image = cv2.resize(image, (128, 128))
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features = hog(gray_image, **hog_params)
    predicted_class = clf.predict([hog_features])[0]
    print("Predicted class:", predicted_class)
else:
    print("Failed to load the image.")