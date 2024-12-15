import os
import cv2
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import glob
import time 


from PIL import Image
from homography_without_opencv import homography_without_opencv
from transformation import *


# Load keypoints from the .mat file
match_data = sio.loadmat('data/kp_gmaps.mat')
# print(match_data.keys())
# print("kp:gmaps:")
# print(match_data['kp_gmaps'])

points_img1 = match_data['kp_gmaps'][:, 0:2]  # Points in img1
points_gmap = match_data['kp_gmaps'][:, 2:4]  # Points in gmap_img

# gmap_img = Image.open('gmaps_alamedaIST.png')
# test_img = Image.open('img_0001.jpg')
# gmap_img.show()
# test_img.show()



x1, y1 = points_img1[:, 0], points_img1[:, 1]
x2, y2 = points_gmap[:, 0], points_gmap[:, 1]
H = homography_without_opencv(x1, y1, x2, y2)
print(f"Homography Matrix : \n {H}")


# # Loop over images from img_0001.jpg to img_0020.jpg
# for i in range(1, 5):
#     # Format the image file name
#     image_path = f'./data/img_{i:04d}.jpg'
#     yolo_mat_file = f'./data/yolo_{i:04d}.mat'
    
#     # Process the image with bounding boxes and transformation
#     transformed_image, _ = draw_bounding_boxes_and_transform(yolo_mat_file, image_path, H)
    
#     # Display the transformed image
#     display_image(transformed_image)
    
#     # Optional: Wait for a short period to simulate video playback
#     time.sleep(0.05)  # Adjust the sleep time for desired frame rate



# # Loop over images from img_0001.jpg to img_0020.jpg
# for i in range(1, 5):
#     # Format the image file name
#     image_path = f'./data/img_{i:04d}.jpg'
#     yolo_mat_file = f'./data/yolo_{i:04d}.mat'
    
#     # Process the image with bounding boxes and transformation
#     transformed_image, _ = draw_bounding_boxes_and_transform(yolo_mat_file, image_path, H)
    
#     # Display the transformed image
#     display_image(transformed_image)
    
#     # Optional: Wait for a short period to simulate video playback
#     time.sleep(0.05)  # Adjust the sleep time for desired frame rate



# Define video output settings
frame_width = 640  # Width of the video frame (adjust as needed)
frame_height = 480  # Height of the video frame (adjust as needed)
frame_rate = 20  # Frames per second (adjust for desired speed)

# Create a VideoWriter object to save the video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec (you can also use 'MJPG' or others)
output_video = cv2.VideoWriter('output_video.avi', fourcc, frame_rate, (frame_width, frame_height))

# Loop over images from img_0001.jpg to img_0020.jpg
for i in range(1, 80):
    # Format the image file name
    image_path = f'./data/img_{i:04d}.jpg'
    yolo_mat_file = f'./data/yolo_{i:04d}.mat'
    
    # Process the image with bounding boxes and transformation
    transformed_image, _ = draw_bounding_boxes_and_transform(yolo_mat_file, image_path, H)
    
    # Resize the image to match the video frame size (if necessary)
    transformed_image_resized = cv2.resize(transformed_image, (frame_width, frame_height))
    
    # Write the image frame to the video
    output_video.write(transformed_image_resized)

    # # Optional: Display the frame during the loop (to monitor progress)
    # cv2.imshow('Transformed Image', transformed_image_resized)
    
    # Wait for 50 milliseconds for the key press (adjust as necessary for frame rate)
    # if cv2.waitKey(50) & 0xFF == ord('q'):  # Press 'q' to exit
    #     break

# Release the video writer and close all OpenCV windows
output_video.release()
cv2.destroyAllWindows()
print("Output Video Saved Successfully")
