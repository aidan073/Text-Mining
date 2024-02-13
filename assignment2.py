import os
import string
import math

import nltk
from transformers import AutoTokenizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

from collections import defaultdict

# might be useful?
from collections import Counter


def read_files_in_directory(directory_path):
    # key: tokens value: their frequency in all songs belonging to a genre
    dic_term_frequency = defaultdict(int)

    for file in os.listdir(directory_path):
        with open(directory_path + file, 'r') as rfile:
            for line in rfile:
                tokens = tokenize(line)
                for token in tokens:
                    dic_term_frequency[token] += 1

    return dic_term_frequency


def tokenize(text):
    text = text.strip()
    text = text.lower()
    tokens = tokenizer.tokenize(text)
    tokens = [word for word in tokens if word.lower() not in stop_words]
    return tokens


def freq_to_prob(dic_term_frequency):
    dic_term_prob = {}

    total_terms = sum(dic_term_frequency.values())
    for term in dic_term_frequency:
        dic_term_prob[term] = 1 + (dic_term_frequency[term] * 1.0 / total_terms)

    return dic_term_prob


def calculate_probability(dic_term_prob, input_text):
    prob = 0.0
    input_text = tokenize(input_text)
    probs = []
    for term in input_text:
        prob += math.log10(dic_term_prob.get(term, 1))
        probs.append((term, dic_term_prob.get(term, 1)))

    print(probs)
    return prob


def main():
    text = """You used to call me on my cell phone
Late night when you need my love
Call me on my cell phone"""
    results = {}

    for genre in os.listdir("Lyrics"):
        dic = read_files_in_directory(f"Lyrics/{genre}/")
        prob = freq_to_prob(dic)
        p = calculate_probability(prob, text)
        results[genre] = p
    sorted_dict = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    for key in sorted_dict:
        print(f"{key}: {sorted_dict[key]}")
    return

main()
