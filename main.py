import cv2
import time
import mediapipe as mp
import numpy as np
import pickle
from matplotlib import pyplot as plt 

from extract_pose.functions import mediapipe_detection, draw_landmarks, extract_keypoints
from mail.functions import add_relatives_info, send_mail_process
from sound.functions import say_something

# add_relatives_info()

model = pickle.load(open('model\LSTM_batch_16_final.sav', 'rb'))

cv2.namedWindow('OpenCV Feed', cv2.WINDOW_NORMAL)
cv2.resizeWindow('OpenCV Feed', 1040, 780) # Phóng to cửa sổ

mp_holistic = mp.solutions.holistic # Hàm trả về vị trí các điểm pose trên cơ thể
cap = cv2.VideoCapture(0)  # Mở Camera 0 

i = 0  # Đếm số lần dự đoán
frame_count = 0  # Đếm số khung hình
start_time = pre_time1 = pre_time2 = time.time()  # Đặt thời gian vào lúc mở Camera
predicted_labels = 'Normal'  # Nhãn dự đoán ban đầu là bình thường
window_size = 12  # Độ dài của chuỗi dữ liệu liên tục
num_feature = 132
time_to_predict = 0  # Mô hình dự đoán sau một khoảng thời gian này
locations = [] # Mảng để đưa dữ liệu vào để dự đoán
fps = 0

print(cap.get(cv2.CAP_PROP_FPS))  # FPS cao nhất Camera có thể quay được)
print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Chiều rộng của Camera)
print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Chiều dài của Camera)

# Xác định độ tin cậy trong việc xác định các điểm pose
with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence  =0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()  # Trả về frame của Camera hiện tại
        current_time = time.time() - start_time  # Thời gian hiện tại
        frame_count += 1  # Đếm số lượng khung hình
        frame = cv2.flip(frame, 1)  # Đảo ngược Camera
        
        frame, results = mediapipe_detection(frame, holistic)  # Xác định các điểm pose
        draw_landmarks(frame, results)  # Vẽ các điểm pose và lưu nó vào biến frame
        
        # Trích xuất vị trí của các điểm 'pose'
        keypoint = extract_keypoints(results) 
        locations.append(keypoint)
        locations_array = np.array(locations)  # Chuyển về numpy để reshape
        locations_array = locations_array.reshape(1, -1, num_feature)  # Phù hợp với đầu vào của mô hình
        locations_array = locations_array[:, -window_size:, :]  # Để lấy ra 20 frame gần nhất
        
        # input LSTM (window_size, num_feature)
        
        # time_to_predict = (1 / fps) * 10  # Thời gian 1 frame xuất hiện * 10 --> 10 frame dự đoán 1 lần
        
        # # Sau một thời gian để xuất hiện số frame nhất định
        if locations_array.shape[1] == window_size: #(time.time() - pre_time1) >= time_to_predict and 
            predict_value = model.predict(locations_array)  # Dự đoán 
            predicted_labels = ["Fall" if value >= 0.5 else "Normal" for value in predict_value]  # chọn ngưỡng
            pre_time1 = time.time()  # Lưu lại thời điểm dự đoán
        if "Fall" in predicted_labels:
            break
            
        if (time.time() - pre_time2) >= 0.5:
            fps = frame_count / (time.time() - pre_time2)  # Số khung hình chia cho thời gian đã tính
            frame_count = 0  # Đặt số khung hình về 0
            pre_time2 = time.time()  # Trả về thời gian hiện tại để tính khoảng thời gian khi thực hiện lại vòng lặp

        # Thêm thông tin thời gian hiện tại và FPS hiện tại
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Time: {current_time:.2f} seconds, FPS: {fps:.2f}, Predict: {predicted_labels}"
        cv2.putText(frame, text, (10, 30), font, 0.5, (0,0,255), 1, cv2.LINE_AA)
        
        cv2.imshow('OpenCV Feed', frame)  # In ra frame hiện tại --> Trong vòng lặp sẽ in ra các frame liên tục

        # Tắt Camera khi nhấn 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cv2.imwrite('mail/fall_image.jpg', frame)   
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) # In frame cuối cùng trước khi tắt Camera   
    cap.release()  # Giải phóng các biến đã lưu cũng như dung lượng để máy đỡ lag
    cv2.destroyAllWindows()  # Đóng các cửa sổ, tab mà OpenCV dùng

if "Fall" in predicted_labels:    
    say_something()
    send_mail_process()
    
    ax = plt.gca()

    # Xóa ticks của hàng và cột
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Hiển thị plot
    plt.show()
