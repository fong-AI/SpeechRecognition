import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from modelcenn import SpeechEmotionRecognizer
import tensorflow as tf

# Cấu hình
import os
import shutil
DATA_DIR = r"d:\py\DBTL\Code\Datasets"
MODEL_DIR = "saved_models"
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, "D:\py\DBTL\Code\emotion_app\emotion_model.keras")  # Đổi sang .h5 truyền thống
BATCH_SIZE = 12
EPOCHS = 50

def load_emodb_dataset():
    """Tải dữ liệu từ thư mục EmoDB"""
    dataset = []
    emotions = {
        'W': 'angry', 'L': 'boredom', 'E': 'disgust', 'A': 'fear',
        'F': 'happy', 'T': 'sad', 'N': 'neutral'
    }
    
    for filename in os.listdir(os.path.join(DATA_DIR, "EmoDB")):
        if not filename.endswith('.wav'):
            continue
        
        # Trích xuất thông tin cảm xúc từ tên file
        emotion_code = filename[5]
        if emotion_code not in emotions:
            continue
            
        emotion = emotions[emotion_code]
        filepath = os.path.join(DATA_DIR, "EmoDB", filename)
        dataset.append((filepath, emotion))
    
    return dataset

def load_crema_dataset():
    """Tải dữ liệu từ thư mục Crema"""
    dataset = []
    emotion_map = {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear',
        'HAP': 'happy', 'SAD': 'sad', 'NEU': 'neutral'
    }
    
    for filename in os.listdir(os.path.join(DATA_DIR, "Crema")):
        if not filename.endswith('.wav'):
            continue
            
        parts = filename.split('_')
        if len(parts) < 3:
            continue
            
        emotion_code = parts[2]
        if emotion_code not in emotion_map:
            continue
            
        emotion = emotion_map[emotion_code]
        filepath = os.path.join(DATA_DIR, "Crema", filename)
        dataset.append((filepath, emotion))
    
    return dataset

def load_savee_dataset():
    """Tải dữ liệu từ thư mục Savee"""
    dataset = []
    emotion_map = {
        'a': 'angry', 'd': 'disgust', 'f': 'fear',
        'h': 'happy', 'n': 'neutral', 'sa': 'sad', 'su': 'surprise'
    }
    
    for filename in os.listdir(os.path.join(DATA_DIR, "Savee")):
        if not filename.endswith('.wav'):
            continue
            
        emotion_code = filename.split('_')[1][0]
        if emotion_code not in emotion_map:
            continue
            
        emotion = emotion_map[emotion_code]
        filepath = os.path.join(DATA_DIR, "Savee", filename)
        dataset.append((filepath, emotion))
    
    return dataset

def load_tess_dataset():
    """Tải dữ liệu từ thư mục Tess"""
    dataset = []
    
    for dirname in os.listdir(os.path.join(DATA_DIR, "Tess")):
        dirpath = os.path.join(DATA_DIR, "Tess", dirname)
        if not os.path.isdir(dirpath):
            continue
            
        for filename in os.listdir(dirpath):
            if not filename.endswith('.wav'):
                continue
                
            # Trích xuất cảm xúc từ tên file
            emotion = filename.split('_')[-1].split('.')[0].lower()
            if 'ps' in emotion:  # Xử lý trường hợp 'ps' (pleasant surprise)
                emotion = 'happy'
                
            filepath = os.path.join(dirpath, filename)
            dataset.append((filepath, emotion))
    
    return dataset

def create_dataset():
    """Tạo dataset tổng hợp từ tất cả các nguồn"""
    print("Đang tải dữ liệu...")
    datasets = [
        load_emodb_dataset(),
        load_crema_dataset(),
        load_savee_dataset(),
        load_tess_dataset()
    ]
    
    # Gộp tất cả dữ liệu
    all_data = []
    for dataset in datasets:
        all_data.extend(dataset)
    
    # Chuyển đổi thành DataFrame
    df = pd.DataFrame(all_data, columns=['filepath', 'emotion'])
    
    # Lọc chỉ lấy các cảm xúc cần thiết
    valid_emotions = ['happy', 'sad', 'angry', 'fear', 'neutral']
    df = df[df['emotion'].isin(valid_emotions)]
    
    # Ánh xạ về nhị phân: positive/negative
    def map_to_binary(emotion):
        if emotion in ['happy', 'neutral']:
            return 'positive'
        return 'negative'
    
    df['label'] = df['emotion'].apply(map_to_binary)
    df['label_id'] = df['label'].map({'negative': 0, 'positive': 1})
    
    print(f"Tổng số mẫu: {len(df)}")
    print("Phân bố nhãn:")
    print(df['label'].value_counts())
    
    return df

class AudioDataset(tf.keras.utils.Sequence):
    """Lớp tạo batch dữ liệu để huấn luyện"""
    def __init__(self, filepaths, labels, batch_size=32, shuffle=True):
        self.filepaths = filepaths
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(filepaths))
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.filepaths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_x = []
        batch_y = []
        
        for i in batch_indices:
            try:
                # Sử dụng model để tiền xử lý âm thanh
                model = SpeechEmotionRecognizer()
                features = model.preprocess_audio(self.filepaths[i])
                batch_x.append(features.numpy())
                batch_y.append(self.labels[i])
            except Exception as e:
                print(f"Lỗi khi xử lý file {self.filepaths[i]}: {e}")
                continue
        
        return np.array(batch_x), np.array(batch_y)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

def train():
    # Tạo dataset
    df = create_dataset()
    
    # Chia tập train/validation
    train_df, val_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df['label_id']
    )
    
    print(f"Số mẫu huấn luyện: {len(train_df)}")
    print(f"Số mẫu kiểm tra: {len(val_df)}")
    
    # Tạo data loaders
    train_dataset = AudioDataset(
        train_df['filepath'].values,
        train_df['label_id'].values,
        batch_size=BATCH_SIZE
    )
    
    val_dataset = AudioDataset(
        val_df['filepath'].values,
        val_df['label_id'].values,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    # Khởi tạo và huấn luyện mô hình
    print("Đang khởi tạo mô hình...")
    model = SpeechEmotionRecognizer()
    model.build_model((model.N_MELS, model.MAX_FRAMES, 3))
    
    # Xóa file model cũ nếu tồn tại
    if os.path.exists(MODEL_SAVE_PATH):
        os.remove(MODEL_SAVE_PATH)
    # Tạo thư mục nếu chưa có
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("Bắt đầu huấn luyện...")
    try:
        history = model.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=EPOCHS,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=5,
                    restore_best_weights=True,
                    verbose=1,
                    mode='max'
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_accuracy',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6,
                    verbose=1,
                    mode='max'
                )
            ]
        )
        # Lưu mô hình tốt nhất sau khi huấn luyện (model đã restore_best_weights)
        model.model.save(MODEL_SAVE_PATH)
    except Exception as e:
        print(f"Lỗi trong quá trình huấn luyện: {e}")
        return
    
    print(f"Huấn luyện hoàn tất! Mô hình đã được lưu vào {MODEL_SAVE_PATH}")
    
    # Lưu lịch sử huấn luyện
    hist_df = pd.DataFrame(history.history)
    hist_csv_file = 'training_history.csv'
    hist_df.to_csv(hist_csv_file, index=False)
    print(f"Lịch sử huấn luyện đã được lưu vào {hist_csv_file}")

if __name__ == "__main__":
    # Kiểm tra GPU
    print("GPU có sẵn:", tf.config.list_physical_devices('GPU'))
    
    # Bắt đầu huấn luyện
    train()
