import os
import jieba
import logging
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

# 减少不必要的日志输出
logging.getLogger("jieba").setLevel(logging.WARNING)


# 从inf.txt读取文件路径
def load_file_paths(inf_path):
    with open(inf_path, 'r', encoding='gb18030') as file:
        # 假设每个文件名都在新的一行
        file_names = file.read().strip().split(',')
        # 构建文件路径列表
        return [f"chinese_dataset/{name.strip()}.txt" for name in file_names]


# 载入多个文本文件
def load_texts(file_paths):
    texts = []
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='gb18030') as file:  # 文件编码为gb18030
                texts.append(file.read())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
    return " ".join(texts)


# 中文分词
def segment_text(text):
    words = jieba.cut(text)
    return [word for word in words if word.strip()]


# 保存词频结果到文本文件
def save_word_frequency(frequencies, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        for word, freq in frequencies.items():
            file.write(f"{word}: {freq}\n")


# 绘制齐夫定律图
def plot_zipf_law(frequencies):
    counts = np.array(sorted(frequencies.values(), reverse=True))
    ranks = np.arange(1, len(counts) + 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, counts, marker="o")

    plt.title('Zipf\'s Law')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')

    plt.show()


if __name__ == "__main__":

    inf_path = 'chinese_dataset/inf.txt'
    file_paths = load_file_paths(inf_path)

    output_path = 'word_frequencies.txt'

    text = load_texts(file_paths)
    words = segment_text(text)
    frequencies = Counter(words)

    save_word_frequency(frequencies, output_path)
    print(f"Word frequencies saved to {output_path}.")

    plot_zipf_law(frequencies)
