import cv2
import numpy as np
import mediapipe as mp
from matplotlib import pyplot as plt 
import os
import pandas as pd
import time
from sklearn import decomposition

mp_holistic = mp.solutions.holistic # Hàm trả về vị trí các điểm pose trên cơ thể
mp_drawing = mp.solutions.drawing_utils # Hàm vẽ các điểm pose đó ra


# Phát hiện ra các điểm pose trên cơ thể
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False   #  Xử lý nhanh hơn              
    results = model.process(image)                 
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results


# Trả về vị trí của các pose
def extract_keypoints(results):
    pose = []
    if results.pose_landmarks is not None:
        for res in results.pose_landmarks.landmark:
            location = np.array([res.x, res.y, res.z, res.visibility])
            # Thêm mảng chứa các giá trị của điểm mốc vào danh sách 'pose'
            pose.append(location)

    # Chuyển danh sách 'pose' thành mảng NumPy và làm phẳng mảng đó
    pose = np.array(pose).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    return pose


# Vẽ các điểm pose đó ra 
def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, 
                              results.pose_landmarks, 
                              mp_holistic.POSE_CONNECTIONS) # Vẽ các điểm chính của cơ thể
    
       
# Lấy ra dữ liệu vị trí của từng frame trong video đó, khi có đường dẫn vào video
def frames_extraction(window_size, data):
    location_video = [] # Tổng của dataset (Số video, Frame, Poselandmark) --> video 1 (Frame, Poselandmark) , video 2 (Frame, Poselandmark),..., video cuối cùng (Frame, Poselandmark)
    for index, video_path in data.iloc[:, 1].items():  # Ngã, ngã, ngã,... bình thường, bình thường, ngã,...
        start_time = time.time()
        
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            location_pose = [] # 12 frame với mỗi frame là 132 điểm trên cơ thể --> 1 video (frame, poselandmark)

            video_reader = cv2.VideoCapture(video_path) # Đọc video từ đường dẫn cho trước

            # Lấy ra tổng số frame của video
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

            # Dùng khi video dài hoặc không đều nhưng muốn cắt ra số lượng frame bằng nhau
            # Nếu window_size = 12, số lượng frame của video là 100 thì cứ 5 frame thì lấy 1 frame
            skip_frames_window = max(int(video_frames_count/window_size), 1)

            for frame_counter in range(window_size): 
            
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window) 
                ret, frame = video_reader.read() # Lấy frame từ video trên
                if not ret:
                    break
                # Trích xuất ra vị trí của các điểm 'pose'
                image, results = mediapipe_detection(frame, holistic)
                keypoint = extract_keypoints(results) # 132 điểm trên cơ thể (poselandmark)
                
                # Nếu frame không tồn hay nói cách khác là đã xử lý xong frame cuối cùng
                # ret = None và frame = False --> thoát khỏi vòng lặp 
                
                location_pose.append(keypoint) 
            
            video_reader.release() # Giải phóng các biến đã lưu cũng như dung lượng để máy đỡ lag

        end_time = time.time()
        run_time = end_time - start_time
        
        index += 1
        
        print(f"Video thứ {index}. Thời gian chạy: {run_time:.2f} giây.") # f là float 2 lấy 2 giá trị thập phân cuối
        
        location_video.append(location_pose)
    location_video = np.array(location_video)
    
    # Encoding biến nhãn thành 0 và 1, one-hot encoding nếu có nhiều nhãn
    label_mapping = {"Normal": 0, "Fall": 1}
    numeric_labels = np.array([label_mapping[label] for label in data.iloc[:, 0]])
    
    return location_video, numeric_labels


# In ra hình ảnh có các điểm pose của từng frame trong video
def show_image(data, window_size):
    for video_path in data: # In hết nó lâu
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            count = 1 # Thứ tự frame của video
            
            video_reader = cv2.VideoCapture(video_path) # Đọc video từ đường dẫn cho trước

            # Lấy ra tổng số frame của video
            video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

            # Dùng khi video dài hoặc không đều nhưng muốn cắt ra số lượng frame bằng nhau
            # Nếu window_size = 12, số lượng frame của video là 100 thì cứ 5 frame thì lấy 1 frame
            skip_frames_window = max(int(video_frames_count/window_size), 1)

            for frame_counter in range(window_size):
                
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
                ret, frame = video_reader.read() # Lấy frame từ video trên
                
                # Vẽ các điểm pose 
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                
                plt.figure(figsize=(8, 6))
                plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.title(f'Frame {count}') # In ra thứ tự 
                plt.show()
                # Nếu frame không tồn hay nói cách khác là đã xử lý xong frame cuối cùng
                # ret = None và frame = False --> thoát khỏi vòng lặp 
                count += 1
                if not ret:
                    break

            video_reader.release() # Giải phóng các biến đã lưu cũng như dung lượng để máy đỡ lag
    

# Lấy ra địa chỉ và nhãn của các video
def extract_link_dataset(data_path):# Đường dẫn đến dataset
    rooms = []
    for item in os.listdir(data_path):
        for root, dirs, files in os.walk(os.path.join(data_path, item)):
            for file in files:
                rooms.append((item, os.path.join(root, file)))

    df_link = pd.DataFrame(data=rooms, columns=['tag', 'video_name'])

    return df_link

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    




