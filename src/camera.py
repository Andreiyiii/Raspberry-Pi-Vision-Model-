import cv2

cam=cv2.VideoCapture(0)

while True:
    ret,frame=cam.read()
    if not ret:
        print("Error,no return from camera")
        break

    cv2.imshow('Testing camera', frame)

    if cv2.waitKey(1) == ord("esc"):  
        break




cam.release()
cv2.destroyAllWindows()    

