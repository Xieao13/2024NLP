import warnings
import numpy as np
import time
import jieba
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from matplotlib.font_manager import FontProperties
import random
from gensim.models.callbacks import CallbackAny2Vec

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 设置中文字体
font_path = 'C:/Windows/Fonts/simhei.ttf'  # 根据你的操作系统调整路径
font_prop = FontProperties(fname=font_path)

# 载入预处理后的文本
def load_processed_text(processed_text_path):
    with open(processed_text_path, 'r', encoding='utf-8') as file:
        text = file.readlines()
    sentences = [line.strip().split() for line in text]
    return sentences

class EpochLogger(CallbackAny2Vec):
    '''用于记录训练过程的回调函数'''
    def __init__(self):
        self.epoch = 0

    def on_epoch_begin(self, model):
        print(f"Epoch #{self.epoch} start")

    def on_epoch_end(self, model):
        print(f"Epoch #{self.epoch} end")
        self.epoch += 1

# 训练Word2Vec模型
def train_word2vec_model(sentences, vector_size=100, window=5, min_count=5, workers=4, epochs=100, alpha=0.025, min_alpha=0.0001):
    epoch_logger = EpochLogger()
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        alpha=alpha,
        min_alpha=min_alpha,
        compute_loss=True,
        callbacks=[epoch_logger]
    )
    return model

# 计算词向量之间的语义距离
def compute_similarity(model, word1, word2):
    try:
        similarity = model.wv.similarity(word1, word2)
        return similarity
    except KeyError as e:
        print(f"词汇 '{e.args[0]}' 不在词汇表中")
        return None

# 词语的聚类
def cluster_words(model, n_clusters=10):
    word_vectors = model.wv.vectors
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(word_vectors)
    clusters = kmeans.predict(word_vectors)
    return clusters, kmeans, word_vectors

# 可视化轮廓系数大于0.75的特定簇中的词向量
def plot_high_silhouette_cluster(model, clusters, cluster_label, word_vectors, show_plot=True, save_path=None,sample_size=500):
    labels = list(model.wv.index_to_key)

    # 获取特定簇的索引
    indices = [i for i, label in enumerate(clusters) if label == cluster_label]
    specific_vectors = word_vectors[indices]
    specific_labels = [labels[i] for i in indices]

    # 计算轮廓系数
    silhouette_vals = silhouette_samples(word_vectors, clusters)
    specific_silhouette_vals = silhouette_vals[indices]

    # 筛选轮廓系数大于0.75的点
    high_silhouette_indices = [i for i, val in zip(indices, specific_silhouette_vals) if val > 0.75]
    high_silhouette_vectors = word_vectors[high_silhouette_indices]
    high_silhouette_labels = [labels[i] for i in high_silhouette_indices]
    # 随机抽样部分数据进行绘制
    if sample_size < len(high_silhouette_vectors):
        sampled_indices = np.random.choice(len(high_silhouette_vectors), sample_size, replace=False)
        high_silhouette_vectors = high_silhouette_vectors[sampled_indices]
        high_silhouette_labels = [high_silhouette_labels[i] for i in sampled_indices]

    print("PCA初始降维开始...")
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(specific_vectors)
    print("PCA初始降维完成")

    print("TSNE降维开始...")
    tsne = TSNE(n_components=2, random_state=0, n_iter=250)
    reduced_vectors = tsne.fit_transform(pca_result)
    print("TSNE降维完成")

    # 标准化
    scaler = MinMaxScaler()
    reduced_vectors = scaler.fit_transform(reduced_vectors)

    plt.figure(figsize=(14, 10))
    for i, label in enumerate(specific_labels):
        x, y = reduced_vectors[i, :]
        plt.scatter(x, y)
        plt.annotate(label, (x, y), fontsize=9, fontproperties=font_prop)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    if save_path:
        plt.savefig(save_path)
    if show_plot:
        plt.show()
    plt.close()

# 计算段落之间的语义关联
def compute_paragraph_similarity(model, paragraph1, paragraph2):
    words1 = [word for word in jieba.cut(paragraph1) if word in model.wv]
    words2 = [word for word in jieba.cut(paragraph2) if word in model.wv]

    if not words1 or not words2:
        print("其中一个段落没有有效的词汇。")
        return None

    vector1 = np.mean([model.wv[word] for word in words1], axis=0)
    vector2 = np.mean([model.wv[word] for word in words2], axis=0)

    similarity = cosine_similarity([vector1], [vector2])
    return similarity[0][0]

# 主函数
def main():
    start_time = time.time()

    processed_text_path = 'processed_text.txt'
    sentences = load_processed_text(processed_text_path)

    print(f"加载和预处理数据耗时: {time.time() - start_time:.2f} 秒")
    start_time = time.time()

    # 训练Word2Vec模型
    model = train_word2vec_model(sentences, vector_size=200, window=7, min_count=5, workers=4, epochs=100, alpha=0.025, min_alpha=0.0001)

    print(f"训练 Word2Vec 模型耗时: {time.time() - start_time:.2f} 秒")
    start_time = time.time()

    # 计算词向量之间的语义距离
    word_pairs = [('杨过', '小龙女'), ('倚天剑', '屠龙刀'), ('师父', '师尊'), ('武当', '少林'),('父亲', '爹爹'),('儿子','女儿')]
    for word1, word2 in word_pairs:
        similarity = compute_similarity(model, word1, word2)
        if similarity is not None:
            print(f"'{word1}' 和 '{word2}' 的语义相似度: {similarity}")

    print(f"计算词向量之间的语义距离耗时: {time.time() - start_time:.2f} 秒")
    start_time = time.time()

    # 词语的聚类
    clusters, kmeans, word_vectors = cluster_words(model)
    print(f"词语的聚类结果: {clusters}")

    print(f"词语聚类耗时: {time.time() - start_time:.2f} 秒")
    start_time = time.time()

    # 可视化轮廓系数大于0.75的特定簇中的词向量
    cluster_label = 0  # 要可视化的簇的标签
    plot_high_silhouette_cluster(model, clusters, cluster_label, word_vectors,
                                 save_path='high_silhouette_cluster_plot.png')

    print(f"可视化特定簇中的词向量耗时: {time.time() - start_time:.2f} 秒")
    start_time = time.time()

    # 计算段落之间的语义关联
    paragraph1 = '杨过和小龙女在古墓中生活了很多年。'
    paragraph2 = '郭靖和黄蓉一直住在襄阳。'
    paragraph_similarity = compute_paragraph_similarity(model, paragraph1, paragraph2)
    if paragraph_similarity is not None:
        print(f"段落之间的语义相似度: {paragraph_similarity}")

    print(f"计算段落之间的语义关联耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
