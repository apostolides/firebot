from imageai.Detection.Custom import CustomObjectDetection, CustomVideoObjectDetection
import os
import cv2
execution_path = os.getcwd()
model_path = "detection_model-ex-33--loss-4.97.h5"
model_config = "detection_config.json"

def detect_from_camera():
    camera = cv2.VideoCapture(0)
    # scale down video for better performance
    camera.set(3,320) # camera width
    camera.set(4,240) # camera height
    camera.set(30, 0.1) #camera fps
    detector = CustomVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, model_path))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, model_config))
    detector.loadModel()
    detected_video_path = detector.detectObjectsFromVideo(camera_input=camera,per_frame_function=forFrame, save_detected_video = False, minimum_percentage_probability=40, log_progress=True, return_detected_frame=True )

def detect_from_video(video_input):
    detector = CustomVideoObjectDetection()
    detector.setModelTypeAsYOLOv3()
    detector.setModelPath(detection_model_path=os.path.join(execution_path, model_path))
    detector.setJsonPath(configuration_json=os.path.join(execution_path, model_config))
    detector.loadModel()
    detected_video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path,video_input),per_frame_function=forFrame, save_detected_video = False, minimum_percentage_probability=40, log_progress=True, return_detected_frame=True )

def forFrame(frame_number, output_array, output_count, returned_frame):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")
    cv2.imshow('image',returned_frame)
    cv2.waitKey(1)

if __name__== "__main__":
    video_input = "video1.mp4"
    detect_from_video(video_input)
    #detect_from_camera()
