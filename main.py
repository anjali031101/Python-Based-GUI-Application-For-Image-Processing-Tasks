import pandas as pd
import numpy as np
import turtle
import pyautogui
import pygame
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from keras.models import model_from_json
import cv2
import copy
from scipy.ndimage import gaussian_filter

import tkinter as tk
import joblib as jb
from keras.preprocessing import image

def goggles():
    import cv2
    import numpy as np
    cap= cv2.VideoCapture(0)
    # Load the face cascade XML file for face detection
    face_cascade = cv2.CascadeClassifier('haarcascase.xml')

    # Load the goggles image with transparency
    goggles_img = cv2.imread('heart.png', cv2.IMREAD_UNCHANGED)

    while True:
        # Read the current frame from the video stream
        status, photo = cap.read()

        if not status:
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Calculate the position and size of the face region
            face_roi = photo[y:y + h, x:x + w]

            # Resize the goggles image to match the dimensions of the face region
            goggles_resized = cv2.resize(goggles_img, (w, h))

            # Extract the alpha channel of the goggles image
            alpha = goggles_resized[:, :, 3] / 255.0

            # Create a mask for the goggles region
            mask = alpha.astype(np.uint8)

            # Apply the mask to remove the goggles region from the face
            bg_removed = cv2.bitwise_and(face_roi, face_roi, mask=(1 - mask))

            # Overlay the resized goggles image onto the face region
            output = bg_removed + cv2.bitwise_and(goggles_resized[:, :, :3], goggles_resized[:, :, :3], mask=mask)

            # Replace the face region with the modified output
            photo[y:y + h, x:x + w] = output

            # Draw a rectangle around the face
            #cv2.rectangle(photo, (x, y), (x + w, y + h), [0, 255, 0], 5)

        # Display the modified frame
        cv2.imshow("Video", photo)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(10) == 13:
            break

    # Release the video capture and close the windows
    cap.release()
    cv2.destroyAllWindows()


def BlurrFace():
#face blur
    import cv2

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascase.xml')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest (face) from the frame
            face = frame[y:y+h, x:x+w]

            # Apply a blur effect to the surrounding area of the face
            blurred = cv2.GaussianBlur(frame, (99, 99), 0)

            # Replace the surrounding area of the face with the blurred frame
            frame[y:y+h, x:x+w] = blurred[y:y+h, x:x+w]

        # Display the result
        cv2.imshow('Face Detection with Blur', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1)==13:
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


def Bunny():
    import cv2
    cap= cv2.VideoCapture(0)
    status ,photo = cap.read() 

    # Load the cascade for face detection
    face_cascade = cv2.CascadeClassifier('haarcascase.xml')

    # Load the bunny face image with an alpha channel
    bunny_face = cv2.imread('bunny.png', cv2.IMREAD_UNCHANGED)

    # Extract the bunny face and the alpha channel
    bunny_face_image = bunny_face[:, :, :3]
    bunny_face_mask = bunny_face[:, :, 3]

    # Initialize the video capture
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the video capture
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Iterate over the detected faces
        for (x, y, w, h) in faces:
            # Resize the bunny face to match the size of the detected face
            resized_bunny_face = cv2.resize(bunny_face_image, (w, h))
            resized_bunny_mask = cv2.resize(bunny_face_mask, (w, h))

            # Apply the bunny face mask to create a region of interest (ROI)
            roi = frame[y:y+h, x:x+w]
            roi_bunny = cv2.bitwise_and(resized_bunny_face, resized_bunny_face, mask=resized_bunny_mask)

            # Add the bunny face to the ROI
            bunny_face_final = cv2.add(roi, roi_bunny)

            # Update the frame with the bunny face
            frame[y:y+h, x:x+w] = bunny_face_final

            # Add text 
            cv2.putText(frame, 'Hey Vimal Sir ', (x, y-10), cv2.FONT_HERSHEY_DUPLEX, 0.9, (203, 192, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Video', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1)==13:
            break

    # Release the video capture
    cap.release()

    # Destroy all windows
    cv2.destroyAllWindows()


def Distance():
    import cv2
    import math

    # Load the Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier('haarcascase.xml')

    cap = cv2.VideoCapture(0)

    # Constants for distance measurement
    KNOWN_DISTANCE = 100  # Define a known distance (in cm) from the camera to the face
    KNOWN_FACE_WIDTH = 15  # Define the width of the face (in cm) at the known distance

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        # Process each detected face
        for (x, y, w, h) in faces:
            # Calculate the distance to the face using the known distance and face width
            face_width_pixels = w
            distance = (KNOWN_FACE_WIDTH * cap.get(3)) / (2 * face_width_pixels * math.tan(cap.get(4) * math.pi / 360))
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Draw the distance on the frame
            cv2.putText(frame, f"Distance: {distance:.2f} cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Display the frame with distance information
        cv2.imshow('Distance Measurement', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1)==13:
            break

    cap.release()
    cv2.destroyAllWindows()


def Cropped():
    import cv2

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascase.xml')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Crop the detected face from the frame
            face = frame[y:y+h, x:x+w]

            # Display the cropped face in a separate window
            cv2.imshow('Detected Face', face)

            # Draw a rectangle around the detected face in the original frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Display the frame with detected faces
        cv2.imshow('Face Detection', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) ==13:
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()


def Happy():
    import turtle
    import pygame

    # Initialize Pygame
    pygame.init()

    # Set up the screen
    screen = turtle.Screen()
    screen.title("Doraemon Drawing and Song")
    screen.bgcolor("white")

    # Set up the turtle
    t = turtle.Turtle()
    t.speed(3)

    # Draw head
    t.penup()
    t.goto(0, -100)
    t.pendown()
    t.circle(100)

    # Draw body
    t.penup()
    t.goto(0, -200)
    t.pendown()
    t.circle(150)

    # Draw eyes
    t.penup()
    t.goto(-40, 30)
    t.pendown()
    t.begin_fill()
    t.circle(15)
    t.end_fill()

    t.penup()
    t.goto(40, 30)
    t.pendown()
    t.begin_fill()
    t.circle(15)
    t.end_fill()

    # Draw mouth
    t.penup()
    t.goto(-40, -10)
    t.pendown()
    t.setheading(-60)
    t.circle(40, 120)

    # Draw Doraemon's song
    t.penup()
    t.goto(-70, 130)
    t.pendown()
    t.write("Cheers to the wonderful 45 days journey💗", font=("Arial", 16, "bold"))

    # Hide the turtle
    t.hideturtle()

    # Play the Doraemon song
    pygame.mixer.music.load("Happy Happy - Cat Meme Song.mp3")
    pygame.mixer.music.play()

    # Exit on screen click
    screen.exitonclick()

    # Quit Pygame
    pygame.quit()
    

def VirtualMouse():
    import cv2
    import numpy as np
    import pyautogui
    from cvzone.HandTrackingModule import HandDetector

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)

    detector = HandDetector(detectionCon=0.8, maxHands=1)

    screen_width, screen_height = pyautogui.size()

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        hands, frame = detector.findHands(frame)

        if hands:
            hand = hands[0]
            lmList, bbox = hand["lmList"], hand["bbox"]
            center_x, center_y = int((bbox[0] + bbox[2]) // 2), int((bbox[1] + bbox[3]) // 2)

            fingers = detector.fingersUp(hand)
            total_fingers = fingers.count(1)

            # Move mouse cursor
            mouse_x = np.interp(center_x, [0, screen_width], [0, screen_width])
            mouse_y = np.interp(center_y, [0, screen_height], [0, screen_height])
            pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)

            # Perform click actions
            if total_fingers == 1:
                pyautogui.mouseDown()

            if total_fingers == 0:
                pyautogui.mouseUp()

        cv2.imshow("Hand Tracking", frame)

        if cv2.waitKey(1) == 13:
            break

    cap.release()
    cv2.destroyAllWindows()


def BackBlur():
    import cv2

    # Load the Haar Cascade classifier for face detection
    face_cascade = cv2.CascadeClassifier('haarcascase.xml')

    # Open the webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Read the frame from the webcam
        ret, frame = cap.read()

        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Apply a blur effect to the background
        blurred = cv2.GaussianBlur(frame, (99, 99), 0)

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the region of interest (face) from the frame
            face = frame[y:y+h, x:x+w]

            # Replace the face region with the original, unblurred face
            blurred[y:y+h, x:x+w] = face

        # Display the result
        cv2.imshow('Background Blur with Visible Face', blurred)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) ==13:
            break

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()



def execute_selected_task():
    selected_task = int(choice_var.get())
    if selected_task == 1:
        goggles()
    elif selected_task ==2:
        BlurrFace()
    elif selected_task ==3:
        Bunny()
    elif selected_task ==4:
        Distance()
    elif selected_task ==5:
        Cropped()
    elif selected_task == 6:
        Happy()
    elif selected_task == 7:
        VirtualMouse()
    elif selected_task == 8:
        BackBlur()
    
    
# Create the Tkinter GUI
root = tk.Tk()
root.title("Task Menu")
window_width = 400
window_height = 300
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width - window_width) / 2)
y_coordinate = int((screen_height - window_height) / 2)
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

# Set the background color of the Tkinter window
root.configure(bg="lightgray")

# Create a label to display the menu options
label = tk.Label(root, text="Team 11 \nSelect a task:", bg="lightgray")
label.pack()
# Create a variable to hold the user's choice
choice_var = tk.StringVar()

# Create check buttons for the user to select a task
tk.Checkbutton(root, text="Goggles", variable=choice_var, onvalue="1", offvalue="", bg="lightgray").pack(anchor=tk.W)
tk.Checkbutton(root, text="Blurr Face", variable=choice_var, onvalue="2", offvalue="", bg="lightgray").pack(anchor=tk.W)
tk.Checkbutton(root, text="Bunny", variable=choice_var, onvalue="3", offvalue="", bg="lightgray").pack(anchor=tk.W)
tk.Checkbutton(root, text="Distance", variable=choice_var, onvalue="4", offvalue="", bg="lightgray").pack(anchor=tk.W)
tk.Checkbutton(root, text="Cropped", variable=choice_var, onvalue="5", offvalue="", bg="lightgray").pack(anchor=tk.W)
tk.Checkbutton(root, text="Happy", variable=choice_var, onvalue="6", offvalue="", bg="lightgray").pack(anchor=tk.W)
tk.Checkbutton(root, text="VirtualMouse", variable=choice_var, onvalue="7", offvalue="", bg="lightgray").pack(anchor=tk.W)
tk.Checkbutton(root, text="BackBlur", variable=choice_var, onvalue="8", offvalue="", bg="lightgray").pack(anchor=tk.W)



# Create a button to execute the selected task
execute_button = tk.Button(root, text="Execute Task", command=execute_selected_task, bg="blue", fg="white")
execute_button.pack()

# Start the Tkinter main loop
root.mainloop()
