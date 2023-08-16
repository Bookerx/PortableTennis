

detector_cfg = "assets/yolo/yolov3-1cls.cfg"
detector_weight = "assets/yolo/last.weights"

#RgbVideoCap = 'rtsp://admin:fyp202020@192.168.8.111:554/Streaming/Channels/101/?transportmode=unicast --input-rtsp-latency=0'
TherVideoCap = 'rtsp://admin:hkuit155@192.168.1.64:554/Streaming/Channels/201/?transportmode=unicast --input-rtsp-latency=0'

pose_weight = "assets/pose/latest.pth"
pose_model_cfg = ""
pose_data_cfg = ""

input_src = "assets/videos/video_input3.mp4"
output_src = ""
out_size = (1080, 720)
show_size = (1080, 720)
show = True

write_json = True
json_path = ""
filter_criterion = {}
