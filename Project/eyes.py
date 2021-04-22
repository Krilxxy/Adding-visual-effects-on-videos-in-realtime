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

print("1. Normal Glasses\n2. Red Lens Sunglasses\n3. Classic Sunglasses\n4. Aviator Sunglasses\n5. Glasses frame(no lens)\n6. Green Lens Glasses\n7. Fox Mask")
usrInput = input("What filter would you like? Choose between 1 and 7: ")

imgPrefix = '/resources'
imgPathFox = '/foxFilter.png'
imgPathSun1 = '/sunglasses1.png'
imgPathSun2 = '/sunglasses2.png'
imgPathSun3 = '/sunglasses3.png'
imgPathSun4 = '/sunglasses4.png'
imgPathSun5 = '/sunglasses5.png'
imgPathSun6 = '/sunglasses6.png'

imgPathFinal = ''
centers = []
if usrInput == '1':
    imgPathFinal = imgPrefix + imgPathSun1
    filterX = 0.27
    filterY = 0.55
elif usrInput == '2':
    imgPathFinal = imgPrefix + imgPathSun2
    filterX = 0.27  # correct
    filterY = 0.55
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
    imgPathFinal = imgPrefix + imgPathSun6
    filterX = 0.27  # correct
    filterY = 0.55
elif usrInput == '7':
    imgPathFinal = imgPrefix + imgPathFox
    filterX = 0.32  # correct
    filterY = -0.2



foxFilter_img = read_transparent_png(scriptPass + imgPathFinal)
# foxFilter_img = cv2.imread(scriptPass + '/resources/sunglasses2.png')
if isinstance(foxFilter_img, NoneType):
    raise IOError('Unable to load the image file: ' +
                  scriptPass + imgPathFinal)

orig_gray_overlay_FoxFilter = cv2.cvtColor(foxFilter_img, cv2.COLOR_BGR2GRAY)
ret, orig_mask = cv2.threshold(
    orig_gray_overlay_FoxFilter, 160, 255, cv2.THRESH_BINARY_INV)
orig_mask_inv = cv2.bitwise_not(orig_mask)

cap = cv2.VideoCapture(0)
while True:
    try:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    except:
        break
    vh, vw = frame.shape[:2]
    vh, vw = int(vh), int(vw)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=1)
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eyeCascade.detectMultiScale(roi_gray)
        centers.clear()
        for (x_eye, y_eye, w_eye, h_eye) in eyes:
            centers.append((x + int(x_eye + 0.3*w_eye),
                            y + int(y_eye + 0.3*h_eye)))
            
            print('outside', type(centers))
            if len(centers) > 1:  # if detect both eyes
                h, w = foxFilter_img.shape[:2]
                # extract the REGION OF INTEREST (ROI) from the image
                eye_distance = abs(centers[1][0] - centers[0][0])
                # overlay filter; the factor 2.12 is customizable depending
                # on the size of the face
                foxFilter_width = 2.4 * eye_distance
                scaling_factor = foxFilter_width / w
                print('inside if ',centers)
                #print(scaling_factor, eye_distance, centers)
                overlay_foxFilter = cv2.resize(
                    foxFilter_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                x = centers[0][0] if centers[0][0] < centers[1][0] else centers[1][0]
            
                # customizable X and Y locations; depends on the size of the face
                x -= int(filterX * overlay_foxFilter.shape[1])  # 0.27
                y += int(filterY * overlay_foxFilter.shape[0])  # 0.55
                h, w = overlay_foxFilter.shape[:2]
                h, w = int(h), int(w)
                # if the value is negative, it changes the dtype to uint
                x = max(x, 0)
                y = max(y, 0)
                h = max(h, 1)
                w = max(w, 1)

              #  if h < 0:
              #      h = h + 2**32
              #  if w < 0:
              #      w = w + 2**32

                frame_roi = frame[y:y+h, x:x+w]
                # convert color image to graysacle and threshold it
                gray_overlay_FoxFilter = cv2.cvtColor(overlay_foxFilter, cv2.COLOR_BGR2GRAY)
                ret, mask = cv2.threshold(
                    gray_overlay_FoxFilter, 160, 255, cv2.THRESH_BINARY_INV)

                # create an inverse mask
                mask_inv = cv2.bitwise_not(mask)

                # resizing the masks
                overlay_foxFilter = cv2.resize(
                    foxFilter_img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                mask = cv2.resize(orig_mask, None, fx=scaling_factor,
                                  fy=scaling_factor, interpolation=cv2.INTER_AREA)

                mask_inv = cv2.resize(
                    orig_mask_inv, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

                try:
                    # use the mask to extract the face mask region of interest
                    masked_face = cv2.bitwise_and(
                        overlay_foxFilter, overlay_foxFilter, mask=mask)
                    # use the inverse mask to get the remaining part of the image
                    masked_frame = cv2.bitwise_and(
                        frame_roi, frame_roi, mask=mask_inv)
                except cv2.error as e:
                    print('Ignoring arithmetic exceptions: ' + str(e))
                    #raise e

                # add the two images to get the final output
                frame[y:y+h, x:x+w] = cv2.add(masked_face, masked_frame)
            else:
                print("eyes not found")
            
            cv2.imshow('Eye Detector', frame)
            c = cv2.waitKey(1)
            if c == 27:
                break

cap.release()
cv2.destroyAllWindows()
