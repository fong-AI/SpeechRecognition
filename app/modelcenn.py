import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras import layers, models, callbacks, optimizers

class CeNNBlock(layers.Layer):
    """Custom CeNN block with skip connections and batch normalization"""
    def __init__(self, filters, use_pool=True, **kwargs):
        super().__init__(**kwargs)
        self.convA = layers.Conv2D(filters, 3, padding='same',
                                 kernel_initializer='he_normal')
        self.convB = layers.Conv2D(filters, 3, padding='same',
                                 kernel_initializer='he_normal')
        self.biasI = self.add_weight(shape=(filters,), 
                                   initializer='zeros', 
                                   trainable=True)
        self.act = layers.LeakyReLU(0.1)
        self.bn = layers.BatchNormalization()
        self.drop = layers.Dropout(0.2)
        self.use_pool = use_pool
        if use_pool:
            self.pool = layers.MaxPooling2D((2, 2))

    def call(self, x, training=False):
        b = self.convB(x)
        a = self.convA(self.act(b))
        y = a + b + self.biasI
        y = self.bn(y, training=training)
        if training:
            y = self.drop(y, training=training)
        if self.use_pool:
            y = self.pool(y)
        return y

class SpeechEmotionRecognizer:
    """Speech Emotion Recognition using CeNN model"""
    
    def __init__(self, sr=16000, duration=3.0, n_mels=128, fft_size=1024, hop_size=128):
        # Audio parameters
        self.SR = sr
        self.DURATION = duration
        self.N_MELS = n_mels
        self.FFT_SIZE = fft_size
        self.HOP_SIZE = hop_size
        self.MAX_FRAMES = int(np.ceil((sr * duration - fft_size) / hop_size)) + 1
        
        # Model parameters
        self.model = None
        self.mel_wts = None
        self.class_names = ['negative', 'positive']
        
        # Initialize mel weights
        self._init_mel_weights()
        
    def _init_mel_weights(self):
        """Initialize mel filterbank weights"""
        self.mel_wts = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=self.N_MELS,
            num_spectrogram_bins=self.FFT_SIZE//2 + 1,
            sample_rate=self.SR,
            lower_edge_hertz=20.0,
            upper_edge_hertz=self.SR/2
        )
    
    def compute_deltas(self, mel):
        """Compute delta and delta-delta features"""
        mel_p = tf.pad(mel, [[0, 0], [1, 1]], mode='SYMMETRIC')
        delta = 0.5 * (mel_p[:, 2:] - mel_p[:, :-2])
        delta_p = tf.pad(delta, [[0, 0], [1, 1]], mode='SYMMETRIC')
        delta2 = 0.5 * (delta_p[:, 2:] - delta_p[:, :-2])
        return delta, delta2
    
    def preprocess_audio(self, audio_path):
        """Preprocess audio file into mel spectrogram with deltas"""
        # Read and decode audio
        audio_bin = tf.io.read_file(audio_path)
        wav, _ = tf.audio.decode_wav(audio_bin, desired_channels=1)
        wav = tf.squeeze(wav, -1)
        
        # Pad or trim audio
        desired = tf.cast(self.SR * self.DURATION, tf.int32)
        length = tf.shape(wav)[0]
        wav = tf.cond(
            length < desired,
            lambda: tf.pad(wav, [[0, desired - length]]),
            lambda: wav[:desired]
        )
        
        # Compute STFT
        stfts = tf.signal.stft(
            wav,
            frame_length=self.FFT_SIZE,
            frame_step=self.HOP_SIZE,
            fft_length=self.FFT_SIZE
        )
        mag = tf.abs(stfts)
        
        # Compute mel spectrogram
        mel_spec = tf.tensordot(mag, self.mel_wts, 1)
        mel_spec = tf.transpose(mel_spec)
        
        # Pad or trim to fixed size
        pad_amt = self.MAX_FRAMES - tf.shape(mel_spec)[1]
        mel_spec = tf.cond(
            pad_amt > 0,
            lambda: tf.pad(mel_spec, [[0, 0], [0, pad_amt]]),
            lambda: mel_spec[:, :self.MAX_FRAMES]
        )
        
        log_mel = tf.math.log(mel_spec + 1e-6)
        
        # Compute deltas
        d1, d2 = self.compute_deltas(log_mel)
        mel_3ch = tf.stack([log_mel, d1, d2], axis=-1)
        mel_3ch.set_shape([self.N_MELS, self.MAX_FRAMES, 3])
        
        # Normalize
        mean = tf.reduce_mean(mel_3ch)
        std = tf.math.reduce_std(mel_3ch)
        mel_3ch = (mel_3ch - mean) / (std + 1e-6)
        
        return mel_3ch
    
    def build_model(self, input_shape):
        """Build the CeNN model architecture"""
        inputs = layers.Input(shape=input_shape, name='mel_input')
        x = layers.BatchNormalization()(inputs)
        
        # CeNN blocks
        x = CeNNBlock(32, use_pool=True)(x)
        x = CeNNBlock(64, use_pool=True)(x)
        x = CeNNBlock(128, use_pool=True)(x)
        x = CeNNBlock(128, use_pool=False)(x)
        
        # LSTM layers
        x = layers.Permute((2, 1, 3))(x)
        x_shape = tf.keras.backend.int_shape(x)
        x = layers.Reshape((x_shape[1], x_shape[2] * x_shape[3]))(x)
        
        x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
        x = layers.Bidirectional(layers.LSTM(32))(x)
        
        # Output layer
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)
        
        # Create model
        model = models.Model(inputs=inputs, outputs=outputs, name='cenn_lstm')
        
        # Compile model
        model.compile(
            optimizer=optimizers.Adam(learning_rate=3e-4),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                tf.keras.metrics.AUC(name='auc')
            ]
        )
        
        self.model = model
        return model
    
    def train(self, train_data, val_data, epochs=50, batch_size=32, callbacks=None):
        """Train the model"""
        if self.model is None:
            self.build_model((self.N_MELS, self.MAX_FRAMES, 3))
        
        # Default callbacks
        if callbacks is None:
            callbacks = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ModelCheckpoint(
                    'best_model.h5',
                    save_best_only=True,
                    monitor='val_auc',
                    mode='max',
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ]
        
        # Train model
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, audio_path):
        """Predict emotion from audio file"""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        # Preprocess audio
        mel = self.preprocess_audio(audio_path)
        mel = tf.expand_dims(mel, axis=0)  # Add batch dimension
        
        # Make prediction
        pred = self.model.predict(mel, verbose=0)[0][0]
        label = 1 if pred >= 0.5 else 0
        
        return {
            'probability': float(pred),
            'class': self.class_names[label],
            'confidence': float(pred) if label == 1 else float(1 - pred)
        }
    
    def save(self, path):
        """Save the model to disk"""
        if self.model is None:
            raise ValueError("No model to save")
        self.model.save(path)
    
    @classmethod
    def load(cls, path, **kwargs):
        """Load a saved model"""
        instance = cls(**kwargs)
        instance.model = models.load_model(
            path,
            custom_objects={'CeNNBlock': CeNNBlock}
        )
        return instance

# Example usage
if __name__ == "__main__":
    # Initialize model
    model = SpeechEmotionRecognizer()
    
    # Build model
    model.build_model((model.N_MELS, model.MAX_FRAMES, 3))
    model.model.summary()
    
    # Note: In a real scenario, you would load and preprocess your dataset here
    # and then call model.train() with the appropriate data