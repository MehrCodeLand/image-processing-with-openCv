import cv2 as cv
import matplotlib.pyplot as plt

# Read the image
img = cv.imread('batman1.jpg')

# Convert the image from BGR to RGB (Matplotlib expects RGB)
img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# Display the image
plt.imshow(img_rgb)
plt.axis('off')  # Hide axis
plt.show()
