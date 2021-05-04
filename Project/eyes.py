import os
import cv2
import numpy as np


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
# Function taken from https://www.linkedin.com/pulse/afternoon-debugging-e-commerce-image-processing-nikhil-rasiwasia/

NoneType = type(None)
scriptPass = os.path.dirname(__file__)
faceCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eyeCascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml")

if faceCascade.empty():
    raise IOError('Unable to load the face cascade classifier xml file.')
if eyeCascade.empty():
    raise IOError('Unable to load the eye cascade classifier xml file.')

print("1. Normal Glasses\n2. Red Lens Sunglasses\n3. Classic Sunglasses\n4. Aviator Sunglasses\n5. Glasses frame(no lens)\n6. Fox Mask")
usrInput = input("What filter would you like? Choose between 1 and 6: ")

imgPrefix = '/resources'
imgPathFox = '/medicalMask.png'
imgPathGas = '/gasMask.png'
imgPathSun1 = '/sunglasses1.png'
imgPathSun2 = '/sunglasses2.png'
imgPathSun3 = '/sunglasses3.png'
imgPathSun4 = '/sunglasses4.png'
imgPathSun5 = '/sunglasses5.png'
imgPathSun6 = '/sunglasses6.png'

imgPathFinal = ''
centers = []
old_x = 0
old_y = 0

if usrInput == '1':
    imgPathFinal = imgPrefix + imgPathSun1
    filterX = 0.27
    filterY = 0.55
elif usrInput == '2':
    imgPathFinal = imgPrefix + imgPathSun2
    filterX = 0.19  # correct
    filterY = -0.1
elif usrInput == '3':
    imgPathFinal = imgPrefix + imgPathSun3
    filterX = 0.27
    filterY = 0.55
elif usrInput == '4':
    imgPathFinal = imgPrefix + imgPathSun4
    filterX = 0.27
    filterY = 0.1
elif usrInput == '5':
    imgPathFinal = imgPrefix + imgPathSun5
    filterX = 0.27  # correct
    filterY = 0.67
elif usrInput == '6':
    imgPathFinal = imgPrefix + imgPathFox
    filterX = 0.30  # correct
    filterY = -0.6#-0.2
elif usrInput == '7':
    imgPathFinal = imgPrefix + imgPathGas
    filterX = 0.27  # correct
    filterY = -0.1    

temp_img= read_transparent_png(scriptPass + imgPathFinal) #cv2.imread(scriptPass + imgPathFinal)
cv2.imshow('lol', temp_img)

filter_img = cv2.imread(scriptPass + imgPathFinal)#read_transparent_png(scriptPass + imgPathFinal)
# filter_img = cv2.imread(scriptPass + '/resources/sunglasses2.png')
if isinstance(filter_img, NoneType):
    raise IOError('Unable to load the image file: ' +
                  scriptPass + imgPathFinal)
cv2.imshow('lol2', filter_img)
orig_gray_filter = cv2.cvtColor(filter_img, cv2.COLOR_BGR2GRAY)
ret, orig_mask = cv2.threshold(
    orig_gray_filter, 135, 255, cv2.THRESH_BINARY)
orig_mask_inv = cv2.bitwise_not(orig_mask)

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_AREA)
    except:
        break
    vh, vw = frame.shape[:2]
    vh, vw = int(vh), int(vw)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6)
    
    for (x, y, w, h) in faces:
        ROI_gray = gray[y:y+h, x:x+w]
        #ROI_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(ROI_gray)
        #print (ROI_color, ROI_gray)
        if len(eyes) == 2:
            centers.clear()
        
        # Extracts the values of X and Y 
            for (x_eye, y_eye, w_eye, h_eye) in eyes:
                centers.append((x + int(x_eye + 0.3*w_eye),
                                y + int(y_eye + 0.3*h_eye)))
        
            print('outside', type(centers))
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
            x = min(centers[0][0], centers[1][0]) #x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
            y = min(centers[0][1], centers[1][1]) #y = centers[0][1] if centers[0][1] < centers[1][1] else centers[1][1]
            print('PT X:',centers[0][0], centers[1][0], x, "\nPT Y: ", centers[0][1], centers[1][1], y)
            # filterX & filterY can be changed to fit face
            x -= int(filterX * overlay_filter.shape[1])  # 0.27
            y += int(filterY * overlay_filter.shape[0])  # 0.55
            h, w = overlay_filter.shape[:2]
            h, w = int(h), int(w)
            # If the value is negative, changes the value to positive
            x = max(x, 0)
            y = max(y, 0)
            h = max(h, 1)
            w = max(w, 1)

            frame_roi = frame[y:y+h, x:x+w]
            # Convert color image to graysacle and threshold it
            #gray_overlay_filter = cv2.cvtColor(overlay_filter, cv2.COLOR_BGR2GRAY)
            #ret, mask = cv2.threshold(
            #    gray_overlay_filter, 180, 255, cv2.THRESH_BINARY_INV)

            # Create an inverse mask
            #mask_inv = cv2.bitwise_not(mask)
            

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
            
            if x >= frame.shape[1] - masked_frame.shape[1]:
                x = old_x
            else:
                old_x = x      
            if y >= frame.shape[0] - masked_frame.shape[0]:
                y = old_y
            else:
                old_y = y
            
            print('FRAME SHAPE LOOK HERE',frame.shape[0], frame.shape[1])

            print((frame[y:y+masked_face.shape[0], x:x+masked_face.shape[1]]).shape, frame.shape, x, y)
            print(masked_face.shape)
            print(masked_frame.shape)
            # add the two images to get the final output
            frame[old_y:old_y+masked_face.shape[0], old_x:old_x+masked_face.shape[1]] = cv2.add(masked_frame, masked_face)
        else:
            print("Missing eyes, not poggers", centers)
            
            
    cv2.imshow('Bootleg Snapchat', frame)
    c = cv2.waitKey(1)
    if c == 30:
        break

cap.release()
cv2.destroyAllWindows()
