import streamlit as st
st.set_page_config(page_title='Telecom Tower Insights', page_icon='🗼', layout="wide") 

from PIL import Image
from ultralytics import YOLO
import cv2
import time
import os
from streamlit_option_menu import option_menu
import torch
import pandas as pd
import shutil
import numpy as np
from datetime import timedelta
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

device = '0' if torch.cuda.is_available() else 'cpu'
save_path = "video.mp4"
output_video_path = 'output_video.avi'
final_video_path = 'final_video.mp4'
csv_path = 'time_vs_altitude.csv'

def predict(frame, model):
    """
    Perform object detection inference using the specified YOLO model.
    Saves outputs to runs/detect/predict with exist_ok=True to prevent directory drift.
    """
    results = model.predict(frame, save=True, project='runs/detect', name='predict', exist_ok=True, show_conf=False, line_width=2, device=device)
    print(f"Predicted using {device}")
    return results

def estimate_height(image, height_est_model, base_width=6):
    """
    Estimate the physical height of a telecom tower from an image:
    1. Detects the tower bounding box using YOLO.
    2. Calculates the pixel-to-meter scale factor using the known base width: scale = base_width / bbox_width.
    3. Multiplies bbox pixel height by scale factor to get raw height.
    4. Applies a perspective correction factor (0.8 / ~80% adjustment) to account for ground-level elevation angle (~60 deg).
    5. Overlays visual indicator lines for the base and vertical span.
    """
    result = height_est_model.predict(image, conf=0.7, device=device)[0]
    confidences = result.boxes.conf
    coodinates_tower = result.boxes.xywh

    if len(confidences) > 0:
        # Get index of highest confidence detection
        ind = int(np.argmax(confidences.cpu().numpy() if hasattr(confidences, 'cpu') else confidences))
        x, y, w, h = map(int, coodinates_tower[ind])
        # Adjust width by removing padding margin, ensuring it remains strictly positive
        w_adj = max(w - 20, 1)
        # Draw base line and vertical height line
        cv2.line(image, (x - w_adj // 2, y + h // 2), (x + w_adj // 2, y + h // 2), (0, 255, 0), 2)  # Base line
        cv2.line(image, (x, y - h // 2), (x, y + h // 2), (0, 255, 0), 2)  # Height line
        # Compute height using scale factor (base_width / w_adj)
        height = h * (base_width / w_adj) 
        # Apply perspective compensation (~80% increase due to ~60 degree elevation view angle from ground)
        adjust_h = height * 0.8  
        height += adjust_h
        return image, round(height, 2)
    return image, 0

def check_construction_status(height, max_height):
    """
    Evaluate construction progress against the design maximum height:
    - If completion is >= 85%, mark as Completed.
    - Otherwise, mark as Under Construction with remaining percentage.
    """
    percentage_completed = (height / max_height) * 100

    if percentage_completed >= 85:
        st.success("Construction Completed!")
    else:
        percentage_left = max(0.0, 100 - percentage_completed)
        st.warning(f"Under Construction ⚠ {percentage_left:.2f}% left to be constructed.")

def load_model():
    """
    Load the tower antenna detection model and return the model and its class label dictionary.
    """
    model = YOLO(r'model/tower_model.pt')
    all_labels = model.names
    print("All labels: ", all_labels)
    return model, all_labels

def convert_avi_to_mp4(avi_file_path, output_name):
    """
    Transcode an AVI video file to H.264/AAC MP4 format using FFmpeg.
    """
    os.system(f"ffmpeg -i {avi_file_path} -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 {output_name}")
    if os.path.exists(avi_file_path):
        os.remove(avi_file_path)
    return True

def initialize_plus(height, width, length_percentage=20):
    """
    Compute coordinates for a center targeting reticle (crosshair) for camera/UAV alignment.
    """
    arm_length_x = int(width * length_percentage / 100)
    arm_length_y = int(height * length_percentage / 100)

    center_x, center_y = width // 2, height // 2
    x1, y1 = center_x - arm_length_x // 2, center_y - arm_length_y // 2
    x2, y2 = center_x + arm_length_x // 2, center_y + arm_length_y // 2
    return ([center_x, center_y], [x1, x2], [y1, y2])

def draw_plus(image, cordinates, color):
    """
    Draw a colored crosshair (plus sign) at specified coordinates on the image.
    """
    c = (0, 255, 0)
    if color == 'yellow':
        c = (255, 255, 0)
    elif color == 'red':
        c = (255, 0, 0)

    ([center_x, center_y], [x1, x2], [y1, y2]) = cordinates
    cv2.line(image, (x1, center_y), (x2, center_y), c, 2)
    cv2.line(image, (center_x, y1), (center_x, y2), c, 2)
    return image

def is_box_in_center(box, plus_coordinates):
    """
    Verify whether the center of the crosshair falls within the bounding box.
    """
    center, _, _ = plus_coordinates
    x1_box, y1_box, x2_box, y2_box = box
    is_within_x = x1_box <= center[0] <= x2_box
    is_within_y = y1_box <= center[1] <= y2_box
    return is_within_x and is_within_y

def format_time(ts):
    """
    Format timestamp in milliseconds to HH:MM:SS.mmm format.
    """
    ts = timedelta(milliseconds=ts)
    hours, reminder = divmod(ts.seconds, 3600)
    minutes, seconds = divmod(reminder, 60)
    milliseconds = ts.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}.{milliseconds:03}" 


def main():
    st.subheader('🗼Telecom Tower Insights')
    st.write('''Our state-of-the-art system excels in detecting objects 
                and accurately estimating tower heights, ensuring precision in telecom towers infrastructure analysis.''')
    st.divider()

    if os.path.exists("runs"):
        shutil.rmtree("runs", ignore_errors=True)

    if os.path.exists(save_path):
        os.remove(save_path)

    if os.path.exists(final_video_path):
        os.remove(final_video_path)

    with st.sidebar:
        selected = option_menu(
            menu_title="Main-Menu",
            options=["Image Analytics"],
            icons=["📷"],
            default_index=0,
            styles={"nav-link-selected":{"background-color":"#0033CC"}}
        )

    if selected == "Image Analytics":
        with st.container():
            selected_nav = option_menu(
                menu_title=None,
                options=["Height Estimation", "Infrastructure Analysis"],
                icons=["patch-check", "bookmark-check"],
                default_index=0,
                orientation="horizontal",
                styles={"nav-link-selected":{"background-color":"#0033CC"}}
            )
            
        if selected_nav == "Height Estimation":
            uploaded_file = st.file_uploader("Upload image or video", type=['.jpeg', '.png', '.jpg'], accept_multiple_files=True)
            base_width = float(st.slider("Please set the width (meter) of the base:", 0.0, 10.0, 5.0, step=0.5))
            if st.button('submit', key='btn-1') and uploaded_file:
                st.subheader("Estimations: ")
                height_est_model = YOLO(r'model/height_est.pt')
                for i, img in enumerate(uploaded_file):
                    with st.spinner("Running..."):  
                        img = Image.open(img) 
                        img = np.array(img)
                        # Rotate if image is in landscape mode to ensure vertical alignment
                        if img.shape[0] < img.shape[1]:
                            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        c1, c2 = st.columns(2, gap="large") 
                        ann_image, height = estimate_height(img, height_est_model, base_width=base_width)
                        with c1:
                            st.image(ann_image, use_container_width=True, channels="BGR")
                        with c2:
                            st.subheader(f"Estimated height: {height} meter.")
                            st.write(f"Note: Base of the tower is considered as {base_width} meter.")
                            if height > 0:
                                # Determine maximum design height based on tower base category
                                if base_width < 5:
                                    max_height = 30
                                elif base_width >= 5 and base_width < 7:
                                    max_height = 40
                                else:
                                    max_height = 60
                                check_construction_status(height, max_height)

                    st.divider()

        if selected_nav == "Infrastructure Analysis":
            with st.container():
                selected_subnav = option_menu(
                    menu_title=None,
                    options=["Machienary Detection", "Antenna Detection"],
                    icons=["gear", "broadcast-pin"],
                    default_index=0,
                    orientation="horizontal",
                    styles={"nav-link-selected":{"background-color":"#0033CC"}}
                )
            
            if selected_subnav == "Machienary Detection":
                uploaded_file = st.file_uploader("Upload image", type=['.jpeg', '.png', '.jpg'], accept_multiple_files=True)
                if st.button('submit', key='btn-2') and uploaded_file:
                    st.subheader("Results: ")
                    tele_obj_model = YOLO(r'model/tele_obj_100.pt')
                    label = tele_obj_model.names
                    for i, img in enumerate(uploaded_file):
                        with st.spinner("Running..."):
                            img = Image.open(img) 
                            img = np.array(img)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            res = predict(img, tele_obj_model)[0] 
                            class_counts = {}
                            obj_detected = res.boxes.cls.tolist()
                            for value in obj_detected:
                                cls_name = label.get(value, f"Class_{int(value)}")
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                            # Generate annotated RGB image directly for robust multi-image display
                            annotated_frame = cv2.cvtColor(res.plot(line_width=2), cv2.COLOR_BGR2RGB)
                            c1, c2 = st.columns(2, gap="large") 
                            with c1:
                                st.image(annotated_frame, caption=f"Prediction for image {i+1}", use_container_width=True)
                            with c2:
                                count_df = pd.DataFrame(list(class_counts.items()), columns=['Detected Objects', 'Count'])
                                if len(count_df) > 0:
                                    st.dataframe(count_df, hide_index=True, use_container_width=True)
                                else:
                                    st.info("No machinery objects detected in this image.")

                        st.divider()

            if selected_subnav == "Antenna Detection":
                uploaded_file = st.file_uploader("Upload image", type=['.jpeg', '.png', '.jpg'], accept_multiple_files=True)
                if st.button('submit', key='btn-3') and uploaded_file:
                    st.subheader("Results: ")
                    antenna_model, label = load_model()
                    for i, img in enumerate(uploaded_file):
                        with st.spinner("Running..."):
                            img = Image.open(img) 
                            img = np.array(img)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            res = predict(img, antenna_model)[0] 
                            class_counts = {}
                            obj_detected = res.boxes.cls.tolist()
                            for value in obj_detected:
                                cls_name = label.get(value, f"Class_{int(value)}")
                                class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

                            # Generate annotated RGB image directly for robust multi-image display
                            annotated_frame = cv2.cvtColor(res.plot(line_width=2), cv2.COLOR_BGR2RGB)
                            c1, c2 = st.columns(2, gap="large") 
                            with c1:
                                st.image(annotated_frame, caption=f"Prediction for image {i+1}", use_container_width=True)
                            with c2:
                                count_df = pd.DataFrame(list(class_counts.items()), columns=['Detected Objects', 'Count'])
                                if len(count_df) > 0:
                                    st.dataframe(count_df, hide_index=True, use_container_width=True)
                                else:
                                    st.info("No antennas detected in this image.")

                        st.divider()


if __name__ == '__main__':
    main()
