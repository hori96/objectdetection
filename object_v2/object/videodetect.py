from detect import CustomVideoObjectDetection
import os
from datetime import datetime

def forFrame(frame_number, output_array, output_count):
    print("FOR FRAME " , frame_number)
    print("Output for each object : ", output_array)
    print("Output count for unique objects : ", output_count)
    print("------------END OF A FRAME --------------")


execution_path = os.getcwd()

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()
video_detector.setModelPath("bestdetection_model-ex-012--loss-0006.868.h5")
video_detector.setJsonPath("best1.json")
video_detector.loadModel()

start_time = datetime.now()
video_detector.detectObjectsFromVideo(input_file_path="testvideos/YUN00037.MOV",
                                          output_file_path=os.path.join(execution_path, "outvideos/832_60frame"),
                                          frames_per_second=30,
                                          minimum_percentage_probability=40,
                                          frame_detection_interval=60,
                                          log_progress=True, per_frame_function = forFrame)
end_time = datetime.now()
print('Duration: {}'.format(end_time - start_time))
