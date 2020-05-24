from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("bestdetection_model-ex-012--loss-0006.868.h5")
detector.setJsonPath("best1.json") 
detector.loadModel()
detections, extracted_objects_array = detector.detectObjectsFromImage(input_image="image-003.png", output_image_path="1.png", extract_detected_objects=True)

for detection, object_path in zip(detections, extracted_objects_array):
    print(object_path)
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
    print("---------------")
