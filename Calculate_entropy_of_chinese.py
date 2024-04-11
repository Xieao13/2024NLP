# -*- coding: utf-8 -*-
import jieba
from collections import Counter
import math
import re


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


# 更新后的预处理文本并分词函数
def preprocess_and_tokenize(text):
    # 删除换行符、分页符等隐藏字符
    text = re.sub(r'\s+', '', text)

    # 删除形如[来源:XXX]的引用标记
    text = re.sub(r'\[来源:[^\]]+\]', '', text)

    # 删除标点符号，包括全角和半角形式
    text = re.sub(r'[．，。？！、；：“”‘’（）《》【】『』「」\[\]…—\-,.?!;:"\'()<>=]', '', text)

    # 使用jieba进行中文分词
    tokens = jieba.lcut(text)
    # print(tokens)
    return tokens


# 构建N-Gram模型
def build_n_gram(tokens, n=2):
    n_grams = [' '.join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return n_grams


# 计算信息熵
def calculate_entropy(n_grams):
    n_gram_counts = Counter(n_grams)
    total_n_grams = sum(n_gram_counts.values())

    # 计算概率分布
    probabilities = [count / total_n_grams for count in n_gram_counts.values()]

    # 计算信息熵
    entropy = -sum(p * math.log(p, 2) for p in probabilities)
    return entropy


# 打印N-Gram模型的统计结果
def print_n_gram_stats(n_grams, n, total_characters):
    n_gram_counts = Counter(n_grams)
    unique_n_grams = len(n_gram_counts)
    top_n_grams = n_gram_counts.most_common(10)

    print(f"{n}-gram模型下处理结果：")
    print(f"词库总词数： {sum(n_gram_counts.values())}")
    print(f"不同词的个数： {unique_n_grams}")
    print(f"出现频率前10的{n}-gram词语：")
    for i, (gram, count) in enumerate(top_n_grams, start=1):
        print(f"{i}.({repr(gram)}, {count})")
    print(f"The estimated entropy for Chinese using a {n}-gram model is: {calculate_entropy(n_grams):.2f} bits.\n")


if __name__ == "__main__":

    # 加载文本数据
    inf_path = 'chinese_dataset/inf.txt'
    file_paths = load_file_paths(inf_path)
    text = load_texts(file_paths)

    # 预处理并分词
    tokens = preprocess_and_tokenize(text)

    total_characters = len(text.replace(" ", ""))  # 计算总字符数，排除空格

    print(f"语料库总字数：{total_characters}\n")

    # 对于1-Gram、2-Gram和3-Gram分别进行处理和输出
    for n in range(1, 4):
        n_grams = build_n_gram(tokens, n)
        print_n_gram_stats(n_grams, n, total_characters)