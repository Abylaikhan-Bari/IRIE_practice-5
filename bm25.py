import string
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from math import log

# Загружаем NLP-модель для лемматизации
nlp = spacy.load("en_core_web_sm")

# Исходные документы
documents = [
    "Natural language processing enables machines to understand human language.",
    "Boolean retrieval is a basic model in information retrieval.",
    "Language models are essential for processing and analyzing text.",
    "Understanding Boolean operators is crucial for search engines."
]

# Функция предобработки текста
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    return [word for word in lemmatized_tokens if word.isalnum() and word not in stop_words]

# Применяем предобработку
processed_documents = [preprocess(doc) for doc in documents]

def compute_idf(corpus):
    N = len(corpus)
    idf = {}
    for doc in corpus:
        for word in set(doc):
            idf[word] = idf.get(word, 0) + 1
    for word, freq in idf.items():
        idf[word] = log((N - freq + 0.5) / (freq + 0.5) + 1)
    return idf

def compute_tf(doc):
    tf = {}
    for word in doc:
        tf[word] = tf.get(word, 0) + 1
    return tf

# Вычисление параметров
doc_lengths = [len(doc) for doc in processed_documents]
avgdl = sum(doc_lengths) / len(doc_lengths)
idf = compute_idf(processed_documents)

def bm25_score(query, doc, idf, k1=1.5, b=0.75):
    tf = compute_tf(doc)
    score = 0
    for term in query:
        if term in doc:
            term_tf = tf[term]
            numerator = term_tf * (k1 + 1)
            denominator = term_tf + k1 * (1 - b + b * (len(doc) / avgdl))
            score += idf.get(term, 0) * (numerator / denominator)
    return score

# Тестирование модели
queries = [
    "natural language processing",
    "Boolean retrieval",
    "models text"
]

for query in queries:
    processed_query = preprocess(query)
    scores = [bm25_score(processed_query, doc, idf) for doc in processed_documents]
    sorted_results = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    print(f"Запрос: {query}")
    for rank, (doc_index, score) in enumerate(sorted_results):
        print(f"{rank + 1}. Документ {doc_index + 1}: {documents[doc_index]} (счет: {score:.4f})")
    print()
