import jieba


# 载入文件路径
def load_file_paths(inf_path):
    with open(inf_path, 'r', encoding='gb18030') as file:
        # 假设每个文件名都以逗号隔开
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


# 载入停顿词文件
def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().strip().split('\n'))
    return stopwords


# 对文本进行预处理：分词、去停顿词
def preprocess_text(text, stopwords):
    words = jieba.cut(text)
    filtered_words = [word for word in words if word not in stopwords and word.strip()]
    return filtered_words


# 保存预处理后的文本
def save_processed_text(processed_text, output_path):
    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(" ".join(processed_text))


# 主函数
def main():
    inf_path = 'chinese_dataset/inf.txt'  # 文件名列表的路径
    stopwords_path = 'cn_stopwords.txt'  # 停顿词文件路径
    output_path = 'processed_text.txt'  # 预处理后文本的输出路径

    file_paths = load_file_paths(inf_path)
    raw_text = load_texts(file_paths)
    stopwords = load_stopwords(stopwords_path)
    processed_text = preprocess_text(raw_text, stopwords)
    save_processed_text(processed_text, output_path)


if __name__ == "__main__":
    main()
