from imageai.Detection.Custom import CustomObjectDetection
import os
import cv2
import numpy

execution_path = os.getcwd()


detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath( os.path.join(execution_path , "./fire.h5"))
detector.setJsonPath(configuration_json=os.path.join(execution_path, "detection_config.json"))
detector.loadModel()

cap = cv2.VideoCapture(0);

while(True):
    ret, frame = cap.read()

    detected_image_array, detections = detector.detectObjectsFromImage(output_type="array", input_type="array", input_image=frame)

    for eachObject in detections:
        print(eachObject["name"] , " : ", eachObject["percentage_probability"], " : ", eachObject["box_points"] )

    #adjust camera/hose depending on box_points(?)
    
    cv2.imshow('detection', detected_image_array)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
