import cv2
import os

image_list_file = 'images.txt'
output_video_file = 'output.avi'

# Read the list of images
with open(image_list_file, 'r') as f:
    image_files = f.read().splitlines()

# Determine the size of the first image
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Initialize the video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(output_video_file, fourcc, 10, (width,height))

# Write each image to the video
for image_file in image_files:
    image = cv2.imread(image_file)
    video.write(image)

# Release the video writer and destroy all windows
cv2.destroyAllWindows()
video.release()
