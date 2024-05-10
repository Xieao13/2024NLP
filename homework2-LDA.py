# -*- coding: utf-8 -*-
import os
import random
import jieba
import gensim
from gensim import corpora, models
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np


# 加载文件路径列表
def load_file_paths(inf_path):
    with open(inf_path, 'r', encoding='gb18030') as file:
        file_names = file.read().strip().split(',')
        return [f"chinese_dataset/{name.strip()}.txt" for name in file_names]


# 抽取每本书的K个字或词为一个段落
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


# LDA模型构建和主题特征提取
def build_lda_model(documents, num_topics):
    vectorizer = CountVectorizer(tokenizer=lambda text: jieba.lcut(text))
    doc_term_matrix = vectorizer.fit_transform(documents)
    corpus = gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)
    dictionary = corpora.Dictionary.from_corpus(corpus,
                                                id2word=dict((id, word) for word, id in vectorizer.vocabulary_.items()))
    lda = models.LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary, passes=15, random_state=42)
    return lda, dictionary, vectorizer


def document_to_lda_features(lda_model, dictionary, vectorizer, document):
    vec = vectorizer.transform([document])
    corpus = gensim.matutils.Sparse2Corpus(vec, documents_columns=False)
    topic_probabilities = np.zeros(lda_model.num_topics)
    document_topics = lda_model.get_document_topics(corpus, minimum_probability=0)
    for topic_info in document_topics:
        for topic, prob in topic_info:
            topic_probabilities[topic] = prob
    return topic_probabilities


if __name__ == '__main__':
    inf_path = 'chinese_dataset/inf.txt'
    file_paths = load_file_paths(inf_path)
    K_values = [20, 100, 500, 1000, 3000]  # Different segment lengths
    num_topics_list = [5, 10, 20, 50]  # Different numbers of topics

    for K in K_values:
        for unit in ['word', 'char']:
            documents, labels = extract_paragraphs(file_paths, K, unit=unit)
            for num_topics in num_topics_list:
                lda_model, dictionary, vectorizer = build_lda_model(documents, num_topics=num_topics)
                features = np.array(
                    [document_to_lda_features(lda_model, dictionary, vectorizer, doc) for doc in documents])
                label_encoder = LabelEncoder()
                labels_encoded = label_encoder.fit_transform(labels)
                classifier = MultinomialNB()
                scores = cross_val_score(classifier, features, labels_encoded, cv=10)
                print(f'K={K}, Unit={unit}, Topics={num_topics}, Average accuracy: {np.mean(scores)}')
