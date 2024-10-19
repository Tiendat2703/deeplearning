from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# Xây dựng mô hình LSTM 
def model_LSTM (window_size,num_feature, batch_size, x_train, y_train):
    model = Sequential()
    model.add(LSTM(units=132, return_sequences=True, input_shape=(window_size, num_feature))) 
    model.add(Dropout(0.2)), # Loại bỏ ngẫu nhiên 20% nơron
    
    model.add(LSTM(units=64, return_sequences=True)) # Lớp ẩn với 64 nơron và trả về chuỗi liên tục 
    model.add(Dropout(0.2)), # Loại bỏ ngẫu nhiên 20% nơron
    
    model.add(LSTM(units=64, return_sequences=False)) # Lớp ẩn với 64 nơron và trả về 1 giá trị từ dữ liệu chuỗi liên tục
    model.add(Dropout(0.2)), # Loại bỏ ngẫu nhiên 20% nơron
    
    model.add(Dense(units=1, activation="sigmoid")) # Lớp đầu ra với 1 nơron
    
    # Chọn một ngưỡng để dừng nếu model không cải thiện, và trả về model tốt nhất
    early_stopping_callback = EarlyStopping(monitor = 'val_loss', # Phụ thuộc vào chỉ số hàm mất mát của validation data
                                            patience = 5, # Sau 10 epoch mà không cải thiện thì ngừng
                                            mode = 'min', # min thì là giảm thiểu hàm mất mát, max thì là tối ưu hóa độ chính xác
                                            min_delta = 0.001, # Ngưỡng tối thiểu để epoch đó được xem là có cải thiện
                                            restore_best_weights = True) # Khôi phục lại trọng số tốt nhất của mô hình
    model.compile(optimizer='adam', # Hàm tối ưu hóa là Adam
                  loss='binary_crossentropy', # Nếu dùng nhiều lớp thì dùng sparse_categorical_crossentropy
                  metrics=["accuracy"]) # Độ chính xác

    model.fit(x = x_train, 
              y = y_train, 
              epochs = 25, 
              batch_size = batch_size, 
              shuffle = True, 
              validation_split = 0.2,
              callbacks = [early_stopping_callback])
    
    return model










