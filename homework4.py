import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
import jieba
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp


def load_file_paths(inf_path):
    with open(inf_path, 'r', encoding='gb18030') as file:
        file_names = file.read().strip().split(',')
        return [f"chinese_dataset/{name.strip()}.txt" for name in file_names]

def extract_paragraphs(file_paths, K, unit='word'):
    all_paragraphs = []
    for file_path in file_paths:
        book_name = os.path.basename(file_path).split('.')[0]
        try:
            with open(file_path, 'r', encoding='gb18030') as file:
                content = file.read()
                if unit == 'word':
                    tokens = list(jieba.cut(content))
                else:
                    tokens = list(content)  # Treat each character as a token
                paragraph_length = max(K, len(tokens) // 1000)
                paragraphs = [''.join(tokens[i:i + paragraph_length]) for i in range(0, len(tokens), paragraph_length)]
                all_paragraphs += [(paragraph, book_name) for paragraph in paragraphs]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    selected_samples = random.sample(all_paragraphs, min(1000, len(all_paragraphs)))
    documents = [sample[0] for sample in selected_samples]
    labels = [sample[1] for sample in selected_samples]
    return documents, labels

inf_path = 'chinese_dataset/inf.txt'
file_paths = load_file_paths(inf_path)
documents, labels = extract_paragraphs(file_paths, 100, unit='word')


# 加载文件路径列表和抽取段落代码
def load_file_paths(inf_path):
    with open(inf_path, 'r', encoding='gb18030') as file:
        file_names = file.read().strip().split(',')
        return [f"chinese_dataset/{name.strip()}.txt" for name in file_names]

def extract_paragraphs(file_paths, K, unit='word'):
    all_paragraphs = []
    for file_path in file_paths:
        book_name = os.path.basename(file_path).split('.')[0]
        try:
            with open(file_path, 'r', encoding='gb18030') as file:
                content = file.read()
                if unit == 'word':
                    tokens = list(jieba.cut(content))
                else:
                    tokens = list(content)  # Treat each character as a token
                paragraph_length = max(K, len(tokens) // 1000)
                paragraphs = [''.join(tokens[i:i + paragraph_length]) for i in range(0, len(tokens), paragraph_length)]
                all_paragraphs += [(paragraph, book_name) for paragraph in paragraphs]
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    selected_samples = random.sample(all_paragraphs, min(1000, len(all_paragraphs)))
    documents = [sample[0] for sample in selected_samples]
    labels = [sample[1] for sample in selected_samples]
    return documents, labels

# 准备数据
inf_path = 'chinese_dataset/inf.txt'
file_paths = load_file_paths(inf_path)
documents, _ = extract_paragraphs(file_paths, 100, unit='word')

# 数据预处理
tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)
word_index = tokenizer.word_index

max_sequence_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# 构建Seq2Seq模型
latent_dim = 256
num_words = len(word_index) + 1

encoder_inputs = Input(shape=(None,))
encoder_embedding = tf.keras.layers.Embedding(num_words, latent_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
decoder_embedding = tf.keras.layers.Embedding(num_words, latent_dim)(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(num_words, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

# 训练模型
encoder_input_data = sequences
decoder_input_data = np.zeros_like(sequences)
decoder_output_data = np.zeros((len(sequences), max_sequence_length, num_words), dtype='float32')

for i, seq in enumerate(sequences):
    for t in range(1, len(seq)):
        decoder_input_data[i, t] = seq[t-1]
        decoder_output_data[i, t, seq[t]] = 1.

model.fit([encoder_input_data, decoder_input_data], decoder_output_data, batch_size=64, epochs=10)

# 保存模型
model.save('seq2seq_model.h5')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Embedding, MultiHeadAttention, LayerNormalization, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# 数据预处理
tokenizer = Tokenizer()
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)
word_index = tokenizer.word_index

max_sequence_length = max([len(seq) for seq in sequences])
sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Transformer模型的构建
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

embed_dim = 32  # 嵌入维度
num_heads = 2  # 注意力头数量
ff_dim = 32  # 前馈网络内部层维度
maxlen = max_sequence_length  # 序列最大长度
vocab_size = len(word_index) + 1  # 词汇表大小

inputs = Input(shape=(maxlen,))
embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
x = embedding_layer(inputs)
transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
x = transformer_block(x)
x = Dense(20, activation="relu")(x)
outputs = Dense(vocab_size, activation="softmax")(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 数据准备
input_data = sequences[:, :-1]
target_data = sequences[:, 1:]

# 训练模型
model.fit(input_data, target_data, batch_size=64, epochs=10)

# 保存模型
model.save('transformer_model.h5')
