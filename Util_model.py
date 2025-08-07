import os
import math
import random 
import pandas as pd
import numpy as np
import datetime as dt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, LayerNormalization, MultiHeadAttention, Embedding, GlobalAveragePooling1D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
from dateutil.parser import parse
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Model as KModel, Sequential
from tensorflow.keras import layers, Model as KModel
import cvxpy as cp
import cvxopt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Layer
import warnings
from tensorflow import keras as K
from tensorflow.keras import layers, regularizers
from typing import Optional


devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  details = tf.config.experimental.get_device_details(gpus[0])
  print("GPU details: ", details)
warnings.filterwarnings('ignore')

# ============================================================
class TemperatureSoftmax(tf.keras.layers.Layer):
    """
    Custom Keras Layer สำหรับ Temperature Scaling Softmax
    """
    def __init__(self, temperature=1.0, **kwargs):
        """
        Constructor ของ Layer

        Args:
            temperature (float): ค่าอุณหภูมิ (T) สำหรับปรับความนุ่มนวล
            **kwargs: arguments อื่นๆ ที่ส่งให้ Layer แม่
        """
        # เรียก constructor ของคลาสแม่ (สำคัญมาก)
        super(TemperatureSoftmax, self).__init__(**kwargs)
        
        # เก็บค่า temperature
        self.temperature = temperature
        
        # ตรวจสอบว่า temperature เป็นค่าบวกเสมอ
        assert self.temperature > 0, "Temperature ต้องเป็นค่าบวกเสมอ"

    def call(self, inputs):
        """
        ส่วนตรรกะการทำงานของ Layer (เทียบเท่ากับ forward ใน PyTorch)
        
        Args:
            inputs: Tensor ที่เป็น output จาก Layer ก่อนหน้า (logits)
            
        Returns:
            Tensor ที่เป็นผลลัพธ์ความน่าจะเป็น
        """
        # นำ inputs (logits) มาหารด้วย temperature
        scaled_logits = inputs / self.temperature
        
        # ใช้ฟังก์ชัน softmax ของ TensorFlow
        return tf.nn.softmax(scaled_logits, axis=-1)

    def get_config(self):
        """
        เมธอดนี้จำเป็นสำหรับการ Save/Load โมเดลที่มี Custom Layer
        """
        config = super(TemperatureSoftmax, self).get_config()
        config.update({
            'temperature': self.temperature
        })
        return config

class LearnableTemperature(tf.keras.layers.Layer):
    """
    Custom Keras Layer ที่เรียนรู้ค่า Temperature (T) ได้เอง
    และนำไปหารค่า Logits ก่อนเข้าฟังก์ชัน Softmax
    """
    def __init__(self, **kwargs):
        """
        Constructor ของ Layer (ไม่มีพารามิเตอร์ temperature อีกต่อไป)
        """
        super(LearnableTemperature, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        สร้างพารามิเตอร์ที่เทรนได้ (trainable weight) สำหรับ Temperature
        """
        # สร้างตัวแปรที่เทรนได้สำหรับเก็บค่า log(T)
        # เริ่มต้นด้วย 0.0 หมายความว่า T เริ่มต้นคือ exp(0.0) = 1.0
        self.log_temperature = self.add_weight(
            name='log_temperature',
            shape=(),  # เป็นค่าเดี่ยว (Scalar)
            initializer=tf.keras.initializers.Constant(0.0),
            trainable=True # สำคัญมาก: ต้องกำหนดให้เทรนได้
        )
        # เรียก build ของคลาสแม่
        super(LearnableTemperature, self).build(input_shape)

    def call(self, inputs):
        """
        ส่วนตรรกะการทำงานของ Layer: หาร logits ด้วย temperature ที่เรียนรู้ได้
        """
        # คำนวณ Temperature จาก log_temperature
        # tf.exp() ช่วยให้ค่า T เป็นบวกเสมอ
        temperature = tf.exp(self.log_temperature)

        # หารค่า inputs (logits) ด้วย Temperature
        # เพิ่ม epsilon เล็กน้อยเพื่อป้องกันการหารด้วยศูนย์
        scaled_logits = inputs / (temperature + 1e-7)

        return scaled_logits

    def get_config(self):
        """
        เมธอดนี้จำเป็นสำหรับการ Save/Load โมเดล
        """
        # ไม่มี arguments พิเศษใน constructor ที่ต้องบันทึก
        config = super(LearnableTemperature, self).get_config()
        return config

class DiscreteStepTemperature(tf.keras.layers.Layer):
    """
    Layer ที่บังคับให้ Temperature ที่เรียนรู้ได้มีค่าเป็นขั้นละ 0.5
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # สร้างตัวแปรต่อเนื่องให้โมเดลเรียนรู้ เริ่มต้นที่ 1.0
        # พร้อมกำหนดเงื่อนไขว่าต้องไม่เป็นลบ (Non-Negative)
        self.continuous_temp = self.add_weight(
            name='continuous_temp',
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True,
            constraint=tf.keras.constraints.NonNeg()
        )

    def call(self, inputs):
        # 1. ปัดค่าต่อเนื่องให้เป็นขั้นละ 0.5
        # เช่น ถ้า continuous_temp = 1.6, (1.6 / 0.5) = 3.2, round(3.2) = 3.0, 3.0 * 0.5 = 1.5
        discrete_temp = tf.round(self.continuous_temp / 0.5) * 0.5
        
        # 2. ป้องกันค่า T เป็น 0 เพื่อไม่ให้เกิดการหารด้วยศูนย์
        # ถ้าปัดแล้วได้ 0 ให้ใช้ค่าต่ำสุดที่ 0.5 แทน
        safe_discrete_temp = tf.maximum(discrete_temp, 0.5)

        # 3. ใช้เทคนิค Straight-Through Estimator (STE)
        # ตอน forward pass จะใช้ค่าที่ปัดเศษแล้ว (safe_discrete_temp)
        # ตอน backward pass จะให้ gradient ไหลผ่านไปที่ continuous_temp โดยตรง
        # เสมือนว่าไม่มีการปัดเศษเกิดขึ้น
        ste_temp = self.continuous_temp + tf.stop_gradient(safe_discrete_temp - self.continuous_temp)

        # 4. นำ Temperature ที่ได้ไปหาร logits
        scaled_logits = inputs / ste_temp
        return scaled_logits

    def get_config(self):
        config = super().get_config()
        return config
# ============================================================

class Time2Vector(Layer):
  def __init__(self, seq_len, **kwargs):
    super(Time2Vector, self).__init__()
    self.seq_len = seq_len

  def build(self, input_shape):
    '''Initialize weights and biases with shape (batch, seq_len)'''
    # initiate 4 matrices, 2 for ω and 2 forφ since we need aω and φ matrix for 
    # both non-periodical (linear) and the periodical (sin) features.
    self.weights_linear = self.add_weight(name='weight_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.bias_linear = self.add_weight(name='bias_linear',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)
    
    self.weights_periodic = self.add_weight(name='weight_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)

    self.bias_periodic = self.add_weight(name='bias_periodic',
                                shape=(int(self.seq_len),),
                                initializer='uniform',
                                trainable=True)


  def call(self, x):
    '''Calculate linear and periodic time features'''

    # Exclude Volume and average across the Open, High, Low, and Close prices, resulting 
    # in the shape (batch_size, seq_len)
    x = tf.math.reduce_mean(x[:,:,:4], axis=-1)

    # calculate the non-periodic (linear) time feature and expand the dimension by 1 again ie. (batch_size, seq_len, 1)
    time_linear = self.weights_linear * x + self.bias_linear # Linear time feature
    time_linear = tf.expand_dims(time_linear, axis=-1) # Add dimension (batch, seq_len, 1)
    
    # repeat for the periodic time feature, also resulting in the same matrix shape. (batch_size, seq_len, 1)
    time_periodic = tf.math.sin(tf.multiply(x, self.weights_periodic) + self.bias_periodic)
    time_periodic = tf.expand_dims(time_periodic, axis=-1) # Add dimension (batch, seq_len, 1)

    # concatenate the linear and periodic time feature. (batch_size, seq_len, 2)
    return tf.concat([time_linear, time_periodic], axis=-1) # shape = (batch, seq_len, 2)
   
  def get_config(self): # Needed for saving and loading model with custom layer
    config = super().get_config().copy()
    config.update({'seq_len': self.seq_len})
    return config
  
# ============================================================
## Original ConvSPE implementation
# class ConvSPE(layers.Layer):

#     def __init__(self, d_model, kernel_size=3, **kwargs):

#         super(ConvSPE, self).__init__(**kwargs)
#         self.d_model = d_model
#         self.kernel_size = kernel_size

#         # Define the 1D convolutional layer.
#         # `padding='same'` ensures the output sequence length matches the input.
#         # The input format (batch, seq_len, features) is standard for Conv1D.
#         self.conv = layers.Conv1D(
#             filters=self.d_model,
#             kernel_size=self.kernel_size,
#             padding='same',
#             activation='relu' # Often a non-linearity is useful here
#         )

#     def call(self, x):

#         # The convolution learns local patterns. We add this as a residual
#         # to the original input to provide positional context.
#         return x + self.conv(x)

#     def get_config(self):
#         """Serializes the layer's configuration."""
#         config = super(ConvSPE, self).get_config()
#         config.update({
#             "d_model": self.d_model,
#             "kernel_size": self.kernel_size,
#         })
#         return config

# ============================================================
# ConvSPE implementation with sinusoidal positional encoding - DS
class ConvSPE(layers.Layer):
    def __init__(self, d_model, kernel_size=3, max_len=5000, **kwargs):
        super(ConvSPE, self).__init__(**kwargs)
        self.d_model = d_model
        self.kernel_size = kernel_size
        self.max_len = max_len
        
        # สร้าง positional encoding แบบ sinusoidal
        self.positional_encoding = self.create_sinusoidal_encoding()
        
        # Convolutional layer สำหรับประมวลผล positional encoding
        self.conv = layers.Conv1D(
            filters=self.d_model,
            kernel_size=self.kernel_size,
            padding='same',  # รักษาความยาว sequence
            use_bias=False,  # ไม่ใช้ bias เพื่อรักษารูปแบบ sinusoidal
            activation=None  # ไม่ใช้ activation function
        )

    def create_sinusoidal_encoding(self):
        """สร้าง sinusoidal positional encoding matrix"""
        position = tf.range(self.max_len, dtype=tf.float32)
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * (-tf.math.log(10000.0) / self.d_model))
        
        sin_encoding = tf.sin(position[:, tf.newaxis] * div_term)
        cos_encoding = tf.cos(position[:, tf.newaxis] * div_term)
        
        # สลับ sine และ cosine สำหรับ dimension คู่และคี่
        encoding = tf.zeros((self.max_len, self.d_model))
        encoding = tf.tensor_scatter_nd_update(encoding, 
                                             indices=tf.constant([[i] for i in range(self.max_len)]),
                                             updates=tf.concat([sin_encoding, cos_encoding], axis=1))
        return encoding

    def call(self, token_embeddings):
        batch_size, seq_len = tf.shape(token_embeddings)[0], tf.shape(token_embeddings)[1]
        
        # 1. ดึง sinusoidal positional encoding สำหรับความยาว sequence ปัจจุบัน
        pos_enc = self.positional_encoding[:seq_len, :]
        pos_enc = tf.expand_dims(pos_enc, 0)  # เพิ่ม batch dimension
        pos_enc = tf.tile(pos_enc, [batch_size, 1, 1])  # ขยายสำหรับทุก batch
        
        # 2. ประมวลผล positional encoding ด้วย convolutional layer
        conv_pos_enc = self.conv(pos_enc)
        
        # 3. รวม positional encoding ที่ประมวลแล้วเข้ากับ token embeddings
        return token_embeddings + conv_pos_enc

    def get_config(self):
        config = super(ConvSPE, self).get_config()
        config.update({
            "d_model": self.d_model,
            "kernel_size": self.kernel_size,
            "max_len": self.max_len,
        })
        return config
# ============================================================

class SineSPE(layers.Layer):
    def __init__(self, d_model, max_len=512, **kwargs):

        super(SineSPE, self).__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        # The positional encoding is created as a constant, non-trainable weight.
        self.sine_encoding = self._generate_sine_encoding()

    def _generate_sine_encoding(self):

        # Create a tensor for positions: [0, 1, ..., max_len-1]
        position = tf.range(self.max_len, dtype=tf.float32)[:, tf.newaxis]
        
        # Calculate the division term for the frequencies
        div_term = tf.exp(tf.range(0, self.d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / self.d_model))
        
        # Calculate sine and cosine values
        sines = tf.sin(position * div_term)
        cosines = tf.cos(position * div_term)
        
        # Interleave sine and cosine values
        # e.g., if d_model=4, [sin0, cos0, sin1, cos1]
        # This is done by creating a tensor of shape (max_len, d_model/2, 2)
        # and then reshaping it to (max_len, d_model).
        encoding = tf.stack([sines, cosines], axis=-1)
        encoding = tf.reshape(encoding, [self.max_len, self.d_model])
        
        return tf.cast(encoding, self.compute_dtype)

    def call(self, x):
        # Get the sequence length from the input tensor's shape
        seq_len = tf.shape(x)[1]
        
        # Slice the pre-computed encoding to match the input sequence length
        # and add it to the input. The encoding is broadcasted across the batch dimension.
        return x + self.sine_encoding[tf.newaxis, :seq_len, :]

    def get_config(self):
        """Serializes the layer's configuration."""
        config = super(SineSPE, self).get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return config

# ============================================================

class TemporalPositionalEncoding(layers.Layer):
    def __init__(self, d_model, max_len=5000, **kwargs):
        """
        Initializes the TemporalPositionalEncoding layer.

        Args:
            d_model (int): The dimensionality of the model.
            max_len (int): The maximum possible sequence length.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        
        # Pre-calculate the positional encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Store it as a non-trainable constant tensor
        self.pe = tf.constant(pe, dtype=tf.float32)

    def call(self, x):
        # Get the sequence length from the input tensor's shape
        seq_len = tf.shape(x)[1]
        
        # Slice the pre-computed encoding and add it to the input.
        # The encoding tensor is shaped (max_len, d_model), we slice it to
        # (seq_len, d_model) and it will be broadcasted across the batch dimension.
        return x + self.pe[:seq_len, :]

    def get_config(self):
        """Serializes the layer's configuration."""
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
        })
        return config

# ============================================================

class LearnablePositionalEncoding(layers.Layer):
    """
    A Keras layer for learnable positional encodings.

    This layer creates a trainable embedding matrix for positions, allowing the
    model to learn the positional information directly from the data.
    """
    def __init__(self, d_model, max_len=1024, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = layers.Dropout(rate=dropout)
        
        # Initialize the positional embedding matrix as a trainable weight.
        # The initializer matches the PyTorch default nn.init.uniform_(-0.02, 0.02).
        self.pe = self.add_weight(
            name="positional_embedding", # Explicitly name the arguments
            shape=(self.max_len, self.d_model), # Explicitly name the arguments
            initializer=tf.keras.initializers.RandomUniform(-0.02, 0.02),
            trainable=True,
        )

    def call(self, x, training=False):
        """Adds the learned positional encoding to the input tensor."""
        seq_len = tf.shape(x)[1]
        # Add the sliced positional embeddings to the input
        x = x + self.pe[:seq_len, :]
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
            "dropout": self.dropout.rate,
        })
        return config

# ============================================================

class AbsolutePositionalEncoding(layers.Layer):
    """
    A Keras layer for standard sinusoidal positional encoding with scaling.
    """
    def __init__(self, d_model, max_len=1024, dropout=0.1, scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = layers.Dropout(rate=dropout)
        self.scale_factor = scale_factor

        # Pre-calculate the sinusoidal encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        # Store as a non-trainable constant tensor
        self.pe = tf.constant(pe * self.scale_factor, dtype=tf.float32)

    def call(self, x, training=False):
        """Adds the fixed positional encoding to the input tensor."""
        seq_len = tf.shape(x)[1]
        x = x + self.pe[tf.newaxis, :seq_len, :]
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
            "dropout": self.dropout.rate,
            "scale_factor": self.scale_factor,
        })
        return config

# ============================================================

class tAPE(layers.Layer):
    """
    A Keras layer for a tuned variant of sinusoidal positional encoding.
    """
    def __init__(self, d_model, max_len=1024, dropout=0.1, scale_factor=1.0, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = layers.Dropout(rate=dropout)
        self.scale_factor = scale_factor

        # Pre-calculate the tuned sinusoidal encoding matrix
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len, dtype=np.float32)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-math.log(10000.0) / d_model))
        
        # Apply the tuning factor to the angle
        tuning_factor = d_model / max_len
        pe[:, 0::2] = np.sin((position * div_term) * tuning_factor)
        pe[:, 1::2] = np.cos((position * div_term) * tuning_factor)
        
        self.pe = tf.constant(pe * self.scale_factor, dtype=tf.float32)

    def call(self, x, training=False):
        """Adds the tuned fixed positional encoding to the input tensor."""
        seq_len = tf.shape(x)[1]
        x = x + self.pe[tf.newaxis, :seq_len, :]
        return self.dropout(x, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_len": self.max_len,
            "dropout": self.dropout.rate,
            "scale_factor": self.scale_factor,
        })
        return config

# ============================================================

def get_positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]      
    depths = np.arange(depth)[np.newaxis, :] / depth 
    angle_rates = 1 / (10000**depths)                
    angle_rads = positions * angle_rates             
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)

# class TUPEMultiHeadAttention(layers.Layer):
#     """
#     Multi-Head Attention layer with Untied Positional Encoding (TUPE).
#     This implementation follows the logic of separating content and position
#     interactions in the attention mechanism.
#     """
#     def __init__(self, d_model, num_heads, max_len=5000, **kwargs):
#         super().__init__(**kwargs)
#         self.d_model = d_model
#         self.num_heads = num_heads
#         self.max_len = max_len
#         if d_model % num_heads != 0:
#             raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
#         self.depth = d_model // num_heads

#         # Standard projections for Q, K, V
#         self.wq = layers.Dense(d_model)
#         self.wk = layers.Dense(d_model)
#         self.wv = layers.Dense(d_model)

#         # Learnable embeddings for relative positions
#         # The range of relative positions is [-max_len+1, max_len-1], so 2*max_len-1 total positions
#         self.pos_emb = layers.Embedding(2 * max_len - 1, self.depth)
        
#         # Final dense layer
#         self.dense = layers.Dense(d_model)

#     def split_heads(self, x, batch_size):
#         """Split the last dimension into (num_heads, depth)."""
#         x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
#         return tf.transpose(x, perm=[0, 2, 1, 3])

#     def _get_relative_positional_bias(self, seq_len):
#         """Calculates the relative positional bias matrix."""
#         # Create relative position indices
#         pos = tf.range(seq_len, dtype=tf.int32)
#         relative_pos = pos[:, tf.newaxis] - pos[tf.newaxis, :] # Shape: (seq_len, seq_len)
        
#         # Shift indices to be non-negative for embedding lookup
#         relative_pos_shifted = relative_pos + self.max_len - 1
        
#         # Lookup embeddings
#         pos_bias = self.pos_emb(relative_pos_shifted) # Shape: (seq_len, seq_len, depth)
        
#         return pos_bias

#     def call(self, v, k, q):
#         batch_size = tf.shape(q)[0]
#         seq_len = tf.shape(q)[1]

#         # Project Q, K, V
#         q = self.wq(q)  # (batch_size, seq_len, d_model)
#         k = self.wk(k)  # (batch_size, seq_len, d_model)
#         v = self.wv(v)  # (batch_size, seq_len, d_model)

#         # Split heads
#         q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len, depth)
#         k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len, depth)
#         v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len, depth)

#         # --- TUPE Core Logic ---
#         # 1. Content-to-Content interaction
#         content_scores = tf.matmul(q, k, transpose_b=True)

#         # 2. Content-to-Position interaction (Positional Bias)
#         # We compute a bias based on the relative positions of keys w.r.t queries
#         pos_bias = self._get_relative_positional_bias(seq_len) # (seq_len, seq_len, depth)
        
#         # einsum is efficient for this kind of operation
#         # q_h: (batch, heads, seq_len_q, depth)
#         # pos_bias_h: (seq_len_q, seq_len_k, depth) -> we want (batch, heads, seq_len_q, seq_len_k)
#         content_pos_scores = tf.einsum('bhqd,qkd->bhqk', q, pos_bias)
        
#         # 3. Combine scores
#         attention_scores = content_scores + content_pos_scores
        
#         # Scale the scores
#         attention_scores /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

#         # Apply softmax
#         attention_weights = tf.nn.softmax(attention_scores, axis=-1)

#         # Multiply by V
#         output = tf.matmul(attention_weights, v)  # (batch_size, num_heads, seq_len, depth)

#         # Concatenate heads and apply final dense layer
#         output = tf.transpose(output, perm=[0, 2, 1, 3])
#         output = tf.reshape(output, (batch_size, -1, self.d_model))
#         output = self.dense(output)

#         return output

class TUPEMultiHeadAttention(layers.Layer):
    """
    Multi-Head Attention layer with Untied Positional Encoding (TUPE)
    - แยก interaction ระหว่าง content-content และ content-position อย่างชัดเจน
    - เพิ่ม learnable scaling factor สำหรับแต่ละ head ตามเอกสารวิจัย
    - ใช้ relative positional encoding แบบไม่ผูกมัดกับ content projection
    """
    def __init__(self, d_model, num_heads, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.max_len = max_len
        
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.depth = d_model // num_heads

        # Projections สำหรับ content (ตัวอักษร)
        self.wq = layers.Dense(d_model)  # Content query projection
        self.wk = layers.Dense(d_model)  # Content key projection
        self.wv = layers.Dense(d_model)  # Content value projection

        # Relative positional embeddings (ตำแหน่งสัมพัทธ์)
        self.pos_emb = layers.Embedding(
            input_dim=2 * max_len - 1,  # [-max_len+1, max_len-1]
            output_dim=self.depth        # ใช้ depth เท่ากับ head size
        )
        
        # Learnable scaling factor สำหรับแต่ละ head (gamma)
        self.gamma = self.add_weight(
            name='gamma',
            shape=(num_heads,),
            initializer='ones',
            trainable=True
        )
        
        # Final projection layer
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split ข้อมูลเป็น multiple heads"""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def _get_relative_positional_bias(self, seq_len):
        """คำนวณ relative positional bias matrix"""
        # สร้างดัชนีตำแหน่งสัมพัทธ์ [i-j]
        pos = tf.range(seq_len, dtype=tf.int32)
        relative_pos = pos[:, tf.newaxis] - pos[tf.newaxis, :]  # (seq_len, seq_len)
        
        # ปรับให้ดัชนีเป็นค่าบวก
        relative_pos_shifted = relative_pos + self.max_len - 1
        
        # ค้นหา embeddings
        pos_bias = self.pos_emb(relative_pos_shifted)  # (seq_len, seq_len, depth)
        return pos_bias

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        # Project content tokens
        q_content = self.wq(q)  # (batch, seq_len, d_model)
        k_content = self.wk(k)  # (batch, seq_len, d_model)
        v_content = self.wv(v)  # (batch, seq_len, d_model)

        # Split heads
        q_heads = self.split_heads(q_content, batch_size)  # (batch, num_heads, seq_len, depth)
        k_heads = self.split_heads(k_content, batch_size)  # (batch, num_heads, seq_len, depth)
        v_heads = self.split_heads(v_content, batch_size)  # (batch, num_heads, seq_len, depth)

        # === TUPE Core Logic ===
        # 1. Content-to-Content interaction
        content_scores = tf.matmul(q_heads, k_heads, transpose_b=True)  # (batch, num_heads, seq_len, seq_len)
        
        # 2. Content-to-Position interaction
        pos_bias = self._get_relative_positional_bias(seq_len)  # (seq_len, seq_len, depth)
        content_pos_scores = tf.einsum('bhqd,qkd->bhqk', q_heads, pos_bias)  # (batch, num_heads, seq_len, seq_len)
        
        # 3. ปรับขนาด positional scores ด้วย learnable gamma (head-specific scaling)
        gamma = tf.reshape(self.gamma, [1, -1, 1, 1])  # (1, num_heads, 1, 1)
        content_pos_scores *= gamma
        
        # 4. รวมคะแนน attention
        attention_scores = content_scores + content_pos_scores
        
        # ปรับขนาด (scaling)
        attention_scores /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        
        # Softmax เพื่อคำนวณ weights
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        
        # 5. คูณ weights กับ values
        output = tf.matmul(attention_weights, v_heads)  # (batch, num_heads, seq_len, depth)
        
        # รวม heads กลับ
        output = tf.transpose(output, perm=[0, 2, 1, 3])  # (batch, seq_len, num_heads, depth)
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final projection
        output = self.dense(output)
        return output

class TUPEMultiHeadAttention_gem(layers.Layer):
    """
    Multi-Head Attention layer with a correct implementation of 
    Transformer with Untied Positional Encoding (TUPE-A).
    This layer implements two core principles from the paper:
    1. Untied Correlations: Computes content-content and position-position
       correlations separately using different projection matrices.
    2. Untied [CLS] Token: Resets the positional correlations related to the
       [CLS] token (at index 0) to learnable parameters.
    """
    def __init__(self, d_model, num_heads, max_len=5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.depth = d_model // num_heads

        # 1. Projection matrices for CONTENT (W^Q, W^K, W^V)
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)

        # 2. Projection matrices for ABSOLUTE POSITION (U^Q, U^K)
        self.uq = layers.Dense(d_model)
        self.uk = layers.Dense(d_model)

        # 3. Absolute positional embedding layer
        self.pos_embedding = layers.Embedding(input_dim=max_len, output_dim=d_model)

        # 4. Learnable parameters for untying [CLS] token (theta)
        # --- FIX: Changed shape from (num_heads, 1, 1) to (1, num_heads, 1, 1) to add batch dimension ---
        self.theta_cls_to_other = self.add_weight(
            name='theta_cls_to_other', shape=(1, self.num_heads, 1, 1), initializer='zeros', trainable=True
        )
        self.theta_other_to_cls = self.add_weight(
            name='theta_other_to_cls', shape=(1, self.num_heads, 1, 1), initializer='zeros', trainable=True
        )
        self.theta_cls_to_cls = self.add_weight(
            name='theta_cls_to_cls', shape=(1, self.num_heads, 1, 1), initializer='zeros', trainable=True
        )
        
        # 5. Final dense layer
        self.dense = layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        seq_len = tf.shape(q)[1]

        # --- Part 1: Content-to-Content Correlation ---
        q_c = self.wq(q)
        k_c = self.wk(k)
        v_c = self.wv(v)

        q_c_h = self.split_heads(q_c, batch_size)
        k_c_h = self.split_heads(k_c, batch_size)
        v_c_h = self.split_heads(v_c, batch_size)

        content_scores = tf.matmul(q_c_h, k_c_h, transpose_b=True)

        # --- Part 2: Position-to-Position Correlation ---
        pos_indices = tf.range(start=0, limit=seq_len, delta=1)
        pos_emb = self.pos_embedding(pos_indices)

        q_p = self.uq(pos_emb)
        k_p = self.uk(pos_emb)

        q_p_h = self.split_heads(q_p[tf.newaxis, ...], 1)
        k_p_h = self.split_heads(k_p[tf.newaxis, ...], 1)
        
        position_scores = tf.matmul(q_p_h, k_p_h, transpose_b=True)
        
        # --- Part 3: Untie [CLS] Token from Positions ---
        pos_scores_cls_to_other = self.theta_cls_to_other * tf.ones_like(position_scores[:, :, 0:1, 1:])
        pos_scores_other_to_cls = self.theta_other_to_cls * tf.ones_like(position_scores[:, :, 1:, 0:1])

        modified_pos_scores_row1 = tf.concat([self.theta_cls_to_cls, pos_scores_cls_to_other], axis=-1)
        other_rows = tf.concat([pos_scores_other_to_cls, position_scores[:, :, 1:, 1:]], axis=-1)
        
        position_scores = tf.concat([modified_pos_scores_row1, other_rows], axis=2)

        # --- Part 4: Combine Scores and Compute Attention ---
        attention_scores = content_scores + position_scores
        attention_scores /= tf.math.sqrt(tf.cast(self.depth, tf.float32))
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        output = tf.matmul(attention_weights, v_c_h)

        output = tf.transpose(output, perm=[0, 2, 1, 3])
        output = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(output)

        return output
    

class DifferentiableProjectionLayer(layers.Layer):
    """
    A custom layer that projects weights onto a feasible set defined by multiple
    linear constraints using a differentiable iterative algorithm (based on
    Dykstra's projection algorithm).

    This layer is fully differentiable and runs on the GPU/TPU.
    """
    def __init__(self,
                 asset_columns,
                 asset_map,
                 asset_lower,
                 asset_upper,
                 port_type=(0, 1),
                 num_iterations=20,
                 **kwargs):
        """
        Initializes the layer with all necessary constraints.

        Args:
            asset_columns (pd.Index): List of asset names.
            asset_map (dict): Maps asset names to asset types.
            asset_lower (dict): Lower bounds for each asset type.
            asset_upper (dict): Upper bounds for each asset type.
            port_type (tuple): Defines portfolio type (e.g., (0, 1) for long-only).
            num_iterations (int): Number of projection iterations.
        """
        super().__init__(**kwargs)
        self.asset_columns = asset_columns
        self.asset_map = asset_map
        self.asset_lower = asset_lower
        self.asset_upper = asset_upper
        self.port_type = port_type
        self.num_iterations = num_iterations
        self.num_assets = len(asset_columns)

        # --- Pre-compute constraint matrices and vectors as TensorFlow constants ---
        self._build_constraints()

    def _build_constraints(self):
        """Pre-computes and stores all constraint definitions as tf.constant."""
        # 1. Individual weight bounds (Box constraints)
        ub = np.full(self.num_assets, 0.30, dtype=np.float32)
        if "SHV" in self.asset_columns:
            ub[self.asset_columns.get_loc("SHV")] = 0.40
        lb = np.full(self.num_assets, self.port_type[0], dtype=np.float32)
        self.box_ub = tf.constant(ub, dtype=tf.float32)
        self.box_lb = tf.constant(lb, dtype=tf.float32)

        # 2. Group constraints (Half-space constraints)
        asset_types = {}
        for asset in self.asset_columns:
            t = self.asset_map.get(asset, "Unknown")
            asset_types.setdefault(t, []).append(asset)

        mats, lbs, ubs = [], [], []
        for t, assets in asset_types.items():
            vec = np.zeros(self.num_assets, dtype=np.float32)
            for a in assets:
                idx = self.asset_columns.get_loc(a)
                vec[idx] = 1.0
            mats.append(vec)
            lbs.append(self.asset_lower.get(t, 0.0))
            ubs.append(self.asset_upper.get(t, 1.0))

        self.group_A = tf.constant(np.vstack(mats), dtype=tf.float32) # Shape: (n_groups, n_assets)
        self.group_lb = tf.constant(lbs, dtype=tf.float32) # Shape: (n_groups,)
        self.group_ub = tf.constant(ubs, dtype=tf.float32) # Shape: (n_groups,)
        self.num_groups = self.group_A.shape[0]

    def _project_to_box(self, w):
        """Projects weights onto the box constraints [lb, ub]."""
        return tf.clip_by_value(w, self.box_lb, self.box_ub)

    def _project_to_simplex_sum(self, w):
        """Projects weights onto the hyperplane sum(w) = 1."""
        # Formula: w_proj = w - (sum(w) - 1) / n
        correction = (tf.reduce_sum(w, axis=-1, keepdims=True) - 1.0) / self.num_assets
        return w - correction

    def _project_to_half_space(self, w, A_row, b, is_upper_bound):
        """Projects weights onto a single half-space constraint A_row @ w <= b or >= b."""
        # A_row shape: (n_assets,), w shape: (batch, n_assets)
        # Aw shape: (batch,)
        Aw = tf.reduce_sum(w * A_row, axis=-1)
        
        if is_upper_bound: # Constraint is Aw <= b
            needs_projection = Aw > b
        else: # Constraint is Aw >= b
            needs_projection = Aw < b
        
        # Only apply projection to rows in the batch that violate the constraint
        # Formula: w_proj = w - (A@w - b) / ||A||^2 * A
        # Since ||A||^2 = sum(A_row^2), we can pre-calculate it.
        norm_sq = tf.reduce_sum(A_row**2)
        correction_magnitude = (Aw - b) / (norm_sq + 1e-9)
        
        # Reshape for broadcasting: (batch, 1) * (n_assets,) -> (batch, n_assets)
        correction_vector = tf.expand_dims(correction_magnitude, -1) * A_row
        
        # Use tf.where to apply correction only where needed
        w_projected = tf.where(tf.expand_dims(needs_projection, -1), w - correction_vector, w)
        return w_projected

    def call(self, inputs):
        """
        Performs the iterative projection using tf.while_loop for graph compatibility.
        """
        # Initial state for the main loop
        iter_count = tf.constant(0)
        weights = inputs

        # Main loop condition
        def main_cond(i, w):
            return i < self.num_iterations

        # Main loop body
        def main_body(i, w):
            # One full projection cycle
            w_proj = self._project_to_box(w)
            w_proj = self._project_to_simplex_sum(w_proj)
            
            # Inner loop for group constraints
            group_idx = tf.constant(0)
            def group_cond(j, w_inner):
                return j < self.num_groups
            def group_body(j, w_inner):
                # Project onto the upper and lower bounds for the current group
                w_updated = self._project_to_half_space(w_inner, self.group_A[j], self.group_ub[j], is_upper_bound=True)
                w_updated = self._project_to_half_space(w_updated, self.group_A[j], self.group_lb[j], is_upper_bound=False)
                return j + 1, w_updated
            
            # Execute the inner loop
            _, w_proj = tf.while_loop(group_cond, group_body, [group_idx, w_proj],
                                      shape_invariants=[group_idx.get_shape(), tf.TensorShape([None, self.num_assets])])

            return i + 1, w_proj

        # Execute the main loop
        _, final_weights = tf.while_loop(main_cond, main_body, [iter_count, weights],
                                         shape_invariants=[iter_count.get_shape(), tf.TensorShape([None, self.num_assets])])
        
        # A final cleanup projection to ensure constraints are met as closely as possible
        final_weights = self._project_to_box(final_weights)
        final_weights = self._project_to_simplex_sum(final_weights)
        
        # Explicitly set shape to help Keras
        final_weights.set_shape(inputs.get_shape())
        return final_weights

    def compute_output_shape(self, input_shape):
        """Specifies the output shape of the layer."""
        return input_shape

    def get_config(self):
        # For model saving/loading
        config = super().get_config()
        config.update({
            "asset_columns": self.asset_columns.tolist(),
            "asset_map": self.asset_map,
            "asset_lower": self.asset_lower,
            "asset_upper": self.asset_upper,
            "port_type": self.port_type,
            "num_iterations": self.num_iterations,
        })
        return config
