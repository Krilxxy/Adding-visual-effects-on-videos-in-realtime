#   @file       application.py
#
#   @brief      Application using Python and OpenCV to detect human features & display visual effects using a digital camera
#
#   @author     Razvan-Darius Purcaras

# Library imports
import os
import cv2
import numpy as np

# Function removing the alpha channel from PNG images,
# taken from https://www.linkedin.com/pulse/afternoon-debugging-e-commerce-image-processing-nikhil-rasiwasia/
def read_transparent_png(filename):
    image_4channel = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    alpha_channel = image_4channel[:, :, 3]
    rgb_channels = image_4channel[:, :, :3]

    # White Background Image
    white_background_image = np.ones_like(rgb_channels, dtype=np.uint8) * 255

    # Alpha factor
    alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
    alpha_factor = np.concatenate(
        (alpha_factor, alpha_factor, alpha_factor), axis=2)

    # Transparent Image Rendered on White Background
    base = rgb_channels.astype(np.float32) * alpha_factor
    white = white_background_image.astype(np.float32) * (1 - alpha_factor)
    final_image = base + white
    return final_image.astype(np.uint8)

NoneType = type(None)
scriptPass = os.path.dirname(__file__)

# Loading the OpenCV Classifiers for human features, face and eyes
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")
if faceCascade.empty():
    raise IOError('Face XML not loaded.')
if eyeCascade.empty():
    raise IOError('Eye XML not loaded.')

print("1. Normal Glasses\n2. Red Lens Sunglasses\n3. Classic Sunglasses\n4. Aviator Sunglasses\n5. Glasses frame(no lens)\n6. Fox Mask\n7. T-Pose Man\n8. Vlad's 'Snek'")
usrInput = input("What filter would you like? Choose between 1 and 6: ")

# Setting up the paths for the overlay images
filterPrefix = '/resources'
filterPathFox = '/foxFilter.png'
filterPathMan = '/tposeMan.png'
filterPathSnak = '/snak.png'
filterPathAniEyes = '/animeEyes.png'
filterPath1 = '/sunglasses1.png'
filterPath2 = '/sunglasses2.png'
filterPath3 = '/sunglasses3.png'
filterPath4 = '/sunglasses4.png'
filterPath5 = '/sunglasses5.png'

imgPathFinal = ''
centers = []
old_x = 0
old_y = 0

# According to the user's input, it chooses the desired filter and sets up the coordinates for it 
if usrInput == '1':
    imgPathFinal = filterPrefix + filterPath1
    filterX = -0.22
    filterY = 0.45
elif usrInput == '2':
    imgPathFinal = filterPrefix + filterPath2
    filterX = -0.22 
    filterY = 0.28
elif usrInput == '3':
    imgPathFinal = filterPrefix + filterPath3
    filterX = -0.22
    filterY = 0.35
elif usrInput == '4':
    imgPathFinal = filterPrefix + filterPath4
    filterX = -0.22
    filterY = 0.35
elif usrInput == '5':
    imgPathFinal = filterPrefix + filterPath5
    filterX = -0.22  
    filterY = 0.30
elif usrInput == '6':
    imgPathFinal = filterPrefix + filterPathFox
    filterX = -0.22  
    filterY = 0.6 #-0.2  
elif usrInput == '7':
    imgPathFinal = filterPrefix + filterPathMan
    filterX = -1.22  
    filterY = -1.4 #-0.2  
elif usrInput == '8':
    imgPathFinal = filterPrefix + filterPathSnak
    filterX = -1.22  
    filterY = -0.4 #-0.2 
elif usrInput == '9':
    imgPathFinal = filterPrefix + filterPathAniEyes
    filterX = -0.22  
    filterY = 0.28 #-0.2

temp_img= cv2.imread(scriptPass + imgPathFinal)

filter_img = read_transparent_png(scriptPass + imgPathFinal)
# Throw error if file not found
if isinstance(filter_img, NoneType):
    raise IOError('Unable to load the image file: ' +
                  scriptPass + imgPathFinal)

# Create the mask and the inverse mask for the filter image
orig_gray_filter = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)
ret, orig_mask = cv2.threshold(
    orig_gray_filter, 220, 255, cv2.THRESH_BINARY_INV)
orig_mask_inv = cv2.bitwise_not(orig_mask)

#cv2.imshow('original image with alpha  channel', temp_img)
#cv2.imshow("Orig gray filter", orig_gray_filter)
#cv2.imshow('Orig mask', orig_mask)
#cv2.imshow("Orig mask inv", orig_mask_inv)


cap = cv2.VideoCapture(0)
# Infinite loop showing every frame
while True:
    try:
        ret, frame = cap.read()
        # Resize frame to the desired size
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    except:
        break
    vh, vw = frame.shape[:2]
    vh, vw = int(vh), int(vw)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=12)
    
    # Extracts the coordinate values for the faces
    for (x, y, w, h) in faces:
        ROI_gray = gray[y:y+h, x:x+w]
        ROI_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(ROI_gray)
        # Making sure the list is empty
        if len(eyes) == 2:
            centers.clear()
        
        # Extracts the coordinate values for the eyes
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                # Extracts the center of each eye
                center = (int(x_eye + 0.3*w_eye), int(y_eye + 0.3*h_eye))
                print("This is the center", center)
                print("This is face X", x, "\nThis is face Y", y)
                
                # Find the center coordinates for the left eye first and appends the right eye
                centers.append((x + center[0],
                                y + center[1]))
                print("This is centers list", centers)
                
        if len(centers) == 2:  # if detect both eyes
            
            h, w = filter_img.shape[:2]
            # Extract the REGION OF INTEREST (ROI) from the image
            # ((Y coordinate x value) - (X coordinate x value)) = eye distance
            eye_dist = abs(centers[1][0] - centers[0][0])
            # Creates the size of the filter
            filter_width = 2.3 * eye_dist
            scaling_factor = filter_width / w
            print('inside if ',centers)
            #print(scaling_factor, eye_dist, centers)
            overlay_filter = cv2.resize(
                filter_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
            # Finds the eyes to the closest top left corner 
            x = min(centers[0][0], centers[1][0]) 
            y = min(centers[0][1], centers[1][1]) 
            print('PT X:',x, "\nPT Y: ", y,centers[0][0],centers[1][0], centers[0][1], centers[1][1])
            # filterX & filterY can be changed to fit face
            x += int(filterX * overlay_filter.shape[1])  
            y -= int(filterY * overlay_filter.shape[0])  
            h, w = overlay_filter.shape[:2]
            h, w = int(h), int(w)
            # If the value is negative, changes the value to positive
            x = max(x, 0)
            y = max(y, 0)
            h = max(h, 1)
            w = max(w, 1)

            # ROI for the filter location
            frame_roi = frame[y:y+h, x:x+w]
            
            # Resizing the masks

            mask = cv2.resize(orig_mask, None, fx=scaling_factor,
                                fy=scaling_factor, interpolation=cv2.INTER_AREA)

            mask_inv = cv2.resize(
                orig_mask_inv, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

            try:
                # extracting the filter's ROI using the mask
                masked_face = cv2.bitwise_and(
                    overlay_filter, overlay_filter, mask=mask)
                # extracting the rest of the image using the inverse mask
                masked_frame = cv2.bitwise_and(
                    frame_roi, frame_roi, mask=mask_inv)
            except cv2.error as e:
                print('Ignoring arithmetic exceptions: ' + str(e))
                #raise error
            
            x = min(x, frame.shape[1] - masked_frame.shape[1])
            y = min(y, frame.shape[0] - masked_frame.shape[0])
            
            # Error validation in order to take care of any out of range pixels

            if x >= frame.shape[1] - masked_frame.shape[1]:
                x = old_x
            else:
                old_x = x      
            if y >= frame.shape[0] - masked_frame.shape[0]:
                y = old_y
            else:
                old_y = y
            
            print(masked_face.shape)
            print(masked_frame.shape)
            # add the two images to get the final output
            frame[old_y:old_y+masked_face.shape[0], old_x:old_x+masked_face.shape[1]] = cv2.add(masked_frame, masked_face)
        else:
            print("Missing eyes", centers)
            
            
    cv2.imshow('Visual Effects 1.0', frame)
    c = cv2.waitKey(1)
    if c == 30:
        break

cap.release()
cv2.destroyAllWindows()

#cv2.imshow('removed alpha channel', filter_img)
#cv2.imshow("Eye detection", frame)
#print(scaling_factor, eye_dist, centers)
