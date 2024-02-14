import math
import os
import re
from nltk.tokenize import word_tokenize

class unigram:
    def read_files_to_dictionaries(self, directory_path):
        """This method will iterate on all the files in the input sub-directories and for each song calculates the term
        frequencies
        @param directory_path: directory path where all the directories of different genre are located
        @return: dictionary with song name as the key and the value as a dictionary (A). (A) is a dictionary with key
        representing the token and value being the frequency in the document. The second return type will give you genre for
        each song."""
        dic_song_term_frequency = {}
        dic_song_genre = {}
        for genre in os.listdir(directory_path):
            path = directory_path + "/" + genre
            for file in os.listdir(path):
                temp_dic = {}
                with open(path + "/" + file, 'r') as rfile:
                    for line in rfile:
                        current_line = line.strip()
                        current_line = current_line.lower()
                        current_line = re.sub(r'[^\w\s]', '', current_line)
                        token_list = word_tokenize(current_line)
                        for token in token_list:
                            # Update frequency for existing term or add a new term with frequency 1
                            temp_dic[token] = temp_dic.get(token, 0) + 1
                song_name = file.split(".")[0]
                dic_song_term_frequency[song_name] = temp_dic
                dic_song_genre[song_name] = genre

        return dic_song_term_frequency, dic_song_genre
    
    def get_TF_values(self, dic_song_term_frequency):
        """
        This method takes in token frequency per song as the input and returns TF per token/song
        @param dic_song_term_frequency: song name as key and dictionary A as value. In A, keys are the tokens with their
        frequencies as the values
        @return: Dictionary with song names as keys, and TF-Values as values. These values are also a dictionary of token as
        the keys and their TFs as values
        """
        dic_tf_per_song = {}
        for song, frequency_dict in dic_song_term_frequency.items():
            total_terms = sum(frequency_dict.values())
            tf_per_token = {token: freq / total_terms for token, freq in frequency_dict.items()} # New dictionary with words as keys, TF as values
            dic_tf_per_song[song] = tf_per_token # Set value to tf_per_token dictionary

        return dic_tf_per_song


    def get_IDF_values(self, dic_song_term_frequency):
        """
        This method calculates the IDF values for each token
        @param dic_song_term_frequency: song name as key and dictionary A as value. In A, keys are the tokens with their
        frequencies as the values
        @return: Dictionary with tokens as the keys and IDF values as values
        """
        dic_idf_values = {}
        num_songs = len(dic_song_term_frequency.keys())

        for song in dic_song_term_frequency:
            for word in dic_song_term_frequency[song]:
                # Update df for existing word or add a new word with df 1
                dic_idf_values[word] = dic_idf_values.get(word, 0) + 1

        for word in dic_idf_values:
            # The values of dic_idf_values are currently df, change them to idf
            dic_idf_values[word] = math.log(num_songs / dic_idf_values[word])
        return dic_idf_values
    
    def get_TFIDF_values(self, tf, idf):
        pass

def main():
    text = """You used to call me on my cell phone
Late night when you need my love
Call me on my cell phone"""
    unigram_model = unigram()
    dic_song_dic_term_count, dic_song_genre = unigram_model.read_files_to_dictionaries("Lyrics")
    dic_song_dic_term_frequency = unigram_model.get_TF_values(dic_song_dic_term_count)
    dic_term_idfs = unigram_model.get_IDF_values(dic_song_dic_term_frequency)

main()

