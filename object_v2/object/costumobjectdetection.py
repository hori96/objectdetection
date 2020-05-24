from detect import CustomObjectDetection
from datetime import datetime


detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("bestdetection_model-ex-012--loss-0006.868.h5")
detector.setJsonPath("best1.json")
detector.loadModel()

start_time = datetime.now()
detections = detector.detectObjectsFromImage(input_image="testimages/test2.png", output_image_path="outimages/size/96_2.png", minimum_percentage_probability=40, rectangle_width=25, text_size=7)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])







