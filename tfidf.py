import math
import re


def read_doc_list(filename="tfidf_docs.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def read_stopwords(filename="stopwords.txt"):
    with open(filename, "r", encoding="utf-8") as file:
        return set(word.strip().lower() for word in file if word.strip())


def remove_links(text):
    return re.sub(r'https?://\S+', '', text)


def clean_text(text):
    text = remove_links(text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(words, stopwords):
    return [word for word in words if word not in stopwords]


def stem_word(word):
    if word.endswith("ing") and len(word) > 3:
        return word[:-3]
    elif word.endswith("ly") and len(word) > 2:
        return word[:-2]
    elif word.endswith("ment") and len(word) > 4:
        return word[:-4]
    return word


def stem_words(words):
    return [stem_word(word) for word in words]


def preprocess_document(filename, stopwords):
    with open(filename, "r", encoding="utf-8") as file:
        text = file.read()

    text = clean_text(text)
    words = text.split()
    words = remove_stopwords(words, stopwords)
    words = stem_words(words)

    output_filename = "preproc_" + filename
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(" ".join(words))

    return words


def compute_term_frequencies(words):
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1

    tf = {}
    total_words = len(words)
    if total_words > 0:
        for word in freq:
            tf[word] = freq[word] / total_words

    return tf


def compute_document_frequencies(all_docs_words):
    df = {}
    for words in all_docs_words:
        unique_words = set(words)
        for word in unique_words:
            df[word] = df.get(word, 0) + 1
    return df


def compute_tfidf(words, total_docs, doc_freq):
    tf = compute_term_frequencies(words)
    tfidf_scores = {}

    for word in tf:
        idf = math.log(total_docs / doc_freq[word]) + 1
        tfidf_scores[word] = round(tf[word] * idf, 2)

    return tfidf_scores


def top_5_words(tfidf_scores):
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: (-x[1], x[0]))
    return sorted_words[:5]


def write_tfidf_file(filename, top_words):
    output_filename = "tfidf_" + filename
    with open(output_filename, "w", encoding="utf-8") as file:
        file.write(str(top_words))


def main():
    stopwords = read_stopwords()
    doc_names = read_doc_list()

    all_docs_words = []

    for doc in doc_names:
        words = preprocess_document(doc, stopwords)
        all_docs_words.append(words)

    total_docs = len(all_docs_words)
    doc_freq = compute_document_frequencies(all_docs_words)

    for i in range(len(doc_names)):
        tfidf_scores = compute_tfidf(all_docs_words[i], total_docs, doc_freq)
        top_words = top_5_words(tfidf_scores)
        write_tfidf_file(doc_names[i], top_words)


if __name__ == "__main__":
    main()