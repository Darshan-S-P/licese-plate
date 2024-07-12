import cv2
import pytesseract

harcascade = "model/haarcascade_russian_plate_number.xml"
image_path = "car.jpeg"  

min_area = 500

# Load the image
img = cv2.imread(image_path)


if img is None:
    print("Error: Could not load image.")
else:
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
    
    detected_texts = []

    for (x, y, w, h) in plates:
        area = w * h   
        
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            
            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)
            
            
            text = pytesseract.image_to_string(img_roi, config='--psm 8')  
            print("Detected Number Plate Text:", text)
            detected_texts.append(text.strip())
            
    cv2.imshow("Result", img)   
    
    # Saving text and image here with s key pressed 
    if cv2.waitKey(0) & 0xFF == ord('s'):
        with open("plates/detected_texts.txt", "w") as file:
            for line in detected_texts:
                file.write(line + "\n")
        cv2.imwrite("plates/scanned_img.jpg", img)
        print("Plate and text saved.")
        
    cv2.waitKey(0)
    cv2.destroyAllWindows()
