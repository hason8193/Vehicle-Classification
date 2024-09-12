import cv2
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Load the image
image_path = r'D:\AAA\train\Bicycle\000550_01.jpg'
image = cv2.imread(image_path)
# Resize the original image to 128x128
resized_image = cv2.resize(image, (128, 128))

# Convert the resized image to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

# Calculate HOG features for the grayscale image
fd, hog_image = hog(gray_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(1, 1), visualize=True)

# Rescale histogram for better display
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# Create a figure with 4 subplots
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

# Plot original image
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')

# Plot resized image (128x128)
axes[1].imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
axes[1].set_title('Resized Image (128x128)')

# Plot grayscale image (128x128)
axes[2].imshow(gray_image, cmap=plt.cm.gray)
axes[2].set_title('Grayscale Image (128x128)')

# Plot HOG image (128x128)
axes[3].imshow(hog_image_rescaled, cmap=plt.cm.gray)
axes[3].set_title('HOG Image (128x128)')

plt.tight_layout()
plt.show()