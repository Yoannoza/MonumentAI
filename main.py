"""Object detection demo with MobileNet SSD.
This model and code are based on
https://github.com/robmarkcole/object-detection-app
"""

import av
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer


ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

model = YOLO("yolov8l.pt")

box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

zone_polygon = (ZONE_POLYGON).astype(int)
zone = sv.PolygonZone(polygon=zone_polygon, )
zone_annotator = sv.PolygonZoneAnnotator(
    zone=zone, 
    color=sv.Color.red(),
    thickness=2,
    text_thickness=4,
    text_scale=2
)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:

    result = model(frame, agnostic_nms=True)[0]
    detections = sv.Detections.from_yolov8(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(
        scene=frame, 
        detections=detections, 
        labels=labels
    )

    zone.trigger(detections=detections)
    frame = zone_annotator.annotate(scene=frame)    

    return frame


webrtc_ctx = webrtc_streamer(
    key="object-detection",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={
        "iceServers": get_ice_servers(),
        "iceTransportPolicy": "relay",
    },
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if st.checkbox("Show the detected labels", value=True):
    if webrtc_ctx.state.playing:
        labels_placeholder = st.empty()
        # NOTE: The video transformation with object detection and
        # this loop displaying the result labels are running
        # in different threads asynchronously.
        # Then the rendered video frames and the labels displayed here
        # are not strictly synchronized.
        while True:
            result = result_queue.get()
            labels_placeholder.table(result)

st.markdown(
    "This demo uses a model and code from "
    "https://github.com/robmarkcole/object-detection-app. "
    "Many thanks to the project."
)