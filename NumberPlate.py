import cv2
import pytesseract
import numpy as np

image = cv2.imread("image3.png")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
filtered = cv2.bilateralFilter(gray, 11, 17, 17)
edges = cv2.Canny(filtered, 170, 200)
contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
number_plate_text = None

for contour in contours:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

    if len(approx) == 4:
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(contour)
        roi = gray[y:y+h, x:x+w]
        number_plate_text = pytesseract.image_to_string(roi, config='--psm 11')
        number_plate_text = number_plate_text.strip()
        break


if number_plate_text:
    print(f"Detected Number Plate: {number_plate_text}")
    cv2.imshow('Output', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Number plate not detected")        
        

