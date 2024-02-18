import math
import os
import re
import copy
from nltk.tokenize import word_tokenize

class unigram:
    def read_files(self, directory_path):
        """This method will iterate on all the files in the input sub-directories and for each genre calculates the 
        # of word appearances
        @param directory_path: directory path where all the directories of different genre are located
        @return: dictionary with genre name as the key and the value as a dictionary (A). (A) is a dictionary with key:
        word value: the # of appearances of the word in the genre."""
        dic_genre_word_appearances = {}
        for genre in os.listdir(directory_path): # for genre in lyrics folder
            path = directory_path + "/" + genre
            temp_dic = {} # inner dictionary with key: term value: appearances in the genre
            for file in os.listdir(path): # for song in genre
                with open(path + "/" + file, 'r') as rfile: 
                    for line in rfile: # for line in song
                        
                        # prepare text for tokenization
                        current_line = line.strip()
                        current_line = current_line.lower()
                        current_line = re.sub(r'[^\w\s]', '', current_line)

                        token_list = word_tokenize(current_line)

                        for token in token_list:
                            # update frequency for existing term or add a new term with frequency 1
                            temp_dic[token] = temp_dic.get(token, 0) + 1
            dic_genre_word_appearances[genre] = temp_dic

        return dic_genre_word_appearances # i.e. {'Blues': {'had': 17, 'a': 56, 'bad': 25}}

class tf_idf_calc:
    def get_TF_values(self, dic_genre_word_appearances):
        """
        This method takes in token frequency per genre as the input and returns TF per token in genre
        @param dic_genre_word_appearances: genre name as key and dictionary A as value. In A, keys are the tokens with their
        frequencies as the values
        @return: Dictionary with genre names as keys, and TF-Values as values. These values are also a dictionary of token as
        the keys and their TFs as values
        """
        dic_genre_term_frequency = {}
        for genre, word_appearances in dic_genre_word_appearances.items(): # for genre, inner dictionary (word: # of appearances in genre)
            total_terms = sum(word_appearances.values())
            term_frequency = {word: math.log10((appearances / total_terms)+1) for word, appearances in word_appearances.items()} # new dictionary with words as keys, TF as values
            dic_genre_term_frequency[genre] = term_frequency # set value to tf_per_token dictionary

        return dic_genre_term_frequency


    def get_IDF_values(self, dic_genre_word_appearances):
        """
        This method calculates the IDF values for each token
        @param dic_genre_word_appearances: genre name as key and dictionary A as value. In A, key: words value: # of times the word appears in a genre
        @return: Dictionary with tokens as the keys and IDF values as values
        """
        dic_idf_values = {}
        num_genres = len(dic_genre_word_appearances.keys())

        for genre in dic_genre_word_appearances:
            for word in dic_genre_word_appearances[genre]: # for words in genre
                # update df for existing word or add a new word with df 1
                dic_idf_values[word] = dic_idf_values.get(word, 0) + 1

        for word in dic_idf_values:
            dic_idf_values[word] = math.log10(num_genres / (dic_idf_values[word])) # the values of dic_idf_values are currently df, change them to idf

        return dic_idf_values
    
    def get_TFIDF_values(self, tf, idf):

        dic_genre_tfidf = copy.deepcopy(tf)
        for inner_dic in dic_genre_tfidf.values():
            for word in inner_dic:
                inner_dic[word] = inner_dic[word] * idf[word]
        return dic_genre_tfidf
    
class tester:
    def classify(self, tfidf, text):
        results = {}

        text = text.strip()
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\n', ' ', text)
        text = text.split()

        for word in text:
            for genre in tfidf:
                #if word in tfidf[genre].keys():
                if genre in results:
                    results[genre] += tfidf[genre].get(word, 0)
                else:
                    results[genre] = tfidf[genre].get(word, 0)
        return max(results, key=results.get)


def main():
    text = """I've had every promise generationalrift, there's anger in my heart
You don't know freedombald it's like, chaingar don't have a clue
If you did you'd find yourselves doing the same gorebor"""
    unigram_model = unigram()
    tf_idf_calculations = tf_idf_calc()
    testing = tester()
    dic_genre_word_count = unigram_model.read_files("Lyrics")
    dic_genre_term_frequencies = tf_idf_calculations.get_TF_values(dic_genre_word_count)
    dic_term_idfs = tf_idf_calculations.get_IDF_values(dic_genre_word_count)
    dic_genre_tfidf = tf_idf_calculations.get_TFIDF_values(dic_genre_term_frequencies, dic_term_idfs)
    classification = testing.classify(dic_genre_tfidf, text)
main()
