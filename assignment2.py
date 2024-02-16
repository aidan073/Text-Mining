import math
import os
import re
from nltk.tokenize import word_tokenize

class unigram:
    def read_files(self, directory_path):
        """This method will iterate on all the files in the input sub-directories and for each song calculates the term
        frequencies
        @param directory_path: directory path where all the directories of different genre are located
        @return: dictionary with song name as the key and the value as a dictionary (A). (A) is a dictionary with key:
        word value: the # of appearances of the word in the document."""
        dic_song_word_appearances = {}
        for genre in os.listdir(directory_path): # for genre in lyrics folder
            path = directory_path + "/" + genre
            for file in os.listdir(path): # for song in genre
                temp_dic = {} # inner dictionary with key: term value: appearances in the song
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

                song_name = file.split(".")[0]
                dic_song_word_appearances[song_name] = temp_dic

        return dic_song_word_appearances # i.e. {'Bad Dream': {'had': 8, 'a': 8, 'bad': 18}}


    def get_TF_values(self, dic_song_word_appearances):
        """
        This method takes in token frequency per song as the input and returns TF per token/song
        @param dic_song_word_appearances: song name as key and dictionary A as value. In A, keys are the tokens with their
        frequencies as the values
        @return: Dictionary with song names as keys, and TF-Values as values. These values are also a dictionary of token as
        the keys and their TFs as values
        """
        dic_song_term_frequency = {}
        for song, word_appearances in dic_song_word_appearances.items(): # for song, inner dictionary (word: # of appearances in song)
            total_terms = sum(word_appearances.values())
            term_frequency = {word: math.log10(appearances / total_terms) for word, appearances in word_appearances.items()} # new dictionary with words as keys, TF as values
            dic_song_term_frequency[song] = term_frequency # set value to tf_per_token dictionary

        return dic_song_term_frequency


    def get_IDF_values(self, dic_song_word_appearances):
        """
        This method calculates the IDF values for each token
        @param dic_song_word_appearances: song name as key and dictionary A as value. In A, key: words value: # of times the word appears in a song
        @return: Dictionary with tokens as the keys and IDF values as values
        """
        dic_idf_values = {}
        num_songs = len(dic_song_word_appearances.keys())

        for song in dic_song_word_appearances:
            for word in dic_song_word_appearances[song]: # for words in song
                # update df for existing word or add a new word with df 1
                dic_idf_values[word] = dic_idf_values.get(word, 0) + 1

        for word in dic_idf_values:
            dic_idf_values[word] = math.log10(num_songs / dic_idf_values[word]) # the values of dic_idf_values are currently df, change them to idf

        return dic_idf_values
    
    def get_TFIDF_values(self, tf, idf):
        pass

def main():
    text = """You used to call me on my cell phone
Late night when you need my love
Call me on my cell phone"""
    unigram_model = unigram()
    dic_song_dic_term_count = unigram_model.read_files("Lyrics")
    dic_term_frequencies = unigram_model.get_TF_values(dic_song_dic_term_count)
    dic_term_idfs = unigram_model.get_IDF_values(dic_song_dic_term_count)

main()
