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
                        token_list = token_prep.get_tokens(line) # tokenize
                        for token in token_list:
                            temp_dic[token] = temp_dic.get(token, 0) + 1 # update frequency for existing term or add a new term with frequency 1

            dic_genre_word_appearances[genre] = temp_dic # add inner dic to outer dic

        return dic_genre_word_appearances # i.e. {'Blues': {'had': 17, 'a': 56, 'bad': 25}}
    
class bigram:
    """This method will iterate on all the files in the input sub-directories and for each genre calculates the 
        # of word appearances
        @param directory_path: directory path where all the directories of different genre are located
        @return: dictionary with genre name as the key and the value as a dictionary (A). (A) is a dictionary with key:
        word value: the # of appearances of the word in the genre."""
    def read_files(self, directory_path):
        dic_genre_pair_appearances = {}
        for genre in os.listdir(directory_path): # for genre in lyrics folder
            path = directory_path + "/" + genre
            temp_dic = {} # inner dictionary with key: term value: appearances in the genre
            for file in os.listdir(path): # for song in genre
                with open(path + "/" + file, 'r') as rfile: 
                    for line in rfile: # for line in song              
                        # prepare text for tokenization
                        token_list = token_prep.get_tokens(line) # tokenize
                        token_list.insert(0, "<s>")
                        token_list.append("</s>")
    
                        if not token_list: # if empty line, skip and go to next line
                            continue

                        #first_entry_key = "<s>" + " " + token_list[0]
                        #temp_dic[first_entry_key] = temp_dic.get(first_entry_key, 0) + 1 # create first pair entry
                        for i in range(len(token_list) - 1):
                            key = token_list[i] + " " + token_list[i + 1]
                            # update frequency for existing term or add a new term with frequency 1
                            temp_dic[key] = temp_dic.get(key, 0) + 1
                        #last_entry_key = token_list[-1] + " " + "</s>" 
                        #temp_dic[last_entry_key] = temp_dic.get(last_entry_key, 0) + 1 # create last
            dic_genre_pair_appearances[genre] = temp_dic

        return dic_genre_pair_appearances # i.e. {'Blues': {'<s> had': 9, ...}}
    
class token_prep:
    """
    This method takes in a line of text and preps it for tokenization and then returns the token list
    @param line: line of text to be tokenized
    @return: token list
    """
    @staticmethod
    def get_tokens(line):
        current_line = line.strip()
        current_line = current_line.lower()
        current_line = re.sub(r'[^\w\s]', '', current_line)
        token_list = word_tokenize(current_line)
        return token_list

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
    @staticmethod
    def classify_unigram(tfidf, text, lamda_setter):
        results = {}

        input = token_prep.get_tokens(text)

        for word in input:
            for genre in tfidf:
                if genre in results:
                    results[genre] += tfidf[genre].get(word, 0)
                else:
                    results[genre] = tfidf[genre].get(word, 0)
        if lamda_setter == False:
            return max(results, key=results.get)
        else:
            return results
    @staticmethod
    def classify_bigram(tfidf, text, lambda_setter):
        results = {}

        text = text.split('/n')

        for line in text:
            line = token_prep.get_tokens(line)
            line.insert(0, "<s>")
            line.append("</s>")
            for i in range(len(line) - 1):
                for genre in tfidf:
                    curr = line[i] + " " + line[i + 1]
                    if genre in results:
                        results[genre] += tfidf[genre].get(curr, 0)
                    else:
                        results[genre] = tfidf[genre].get(curr, 0)
        if lambda_setter == False:
            return max(results, key=results.get)
        else:
            return results
    
class lambdas:
    def set_lambda(self, uni_dic_tfidf, bi_dic_tfidf, directory_path):
        most_correct_predictions = 0
        best_lambda = 0
        for i in range(11):
            lambda_value = i / 10.0
            curr_correct_predictions = 0
            for file in os.listdir(directory_path): # for song in validation set
                with open(directory_path + "/" + file, 'r') as rfile: 
                    actual_genre = rfile.readline()
                    actual_genre = actual_genre.strip()
                    text = rfile.read()
                    unigram_results = tester.classify_unigram(uni_dic_tfidf, text, True)
                    bigram_results = tester.classify_bigram(bi_dic_tfidf, text, True)
                    combined_results = {}
                    for genre in unigram_results:
                        combined = lambda_value * unigram_results[genre] + (1 - lambda_value) * bigram_results[genre]
                        combined_results[genre] = combined
                    
                    if max(combined_results, key=combined_results.get) == actual_genre:
                        curr_correct_predictions+=1
            # check if new most_correct_predictions
            if most_correct_predictions < curr_correct_predictions:
                most_correct_predictions = curr_correct_predictions
                best_lambda = lambda_value     
        return best_lambda

def main():
    text = "You used to call me on my cell phone/nLate night when you need my love/nCall me on my cell phone"

    # create instances of objects
    unigram_model = unigram()
    bigram_model = bigram()
    tf_idf_calculations = tf_idf_calc()
    lambdas1 = lambdas()

    # get token counts in genres
    dic_genre_pair_count = bigram_model.read_files("Lyrics")
    dic_genre_word_count = unigram_model.read_files("Lyrics")

    # calculate tfidf scores for unigram
    uni_dic_tf = tf_idf_calculations.get_TF_values(dic_genre_word_count)
    uni_dic_idf = tf_idf_calculations.get_IDF_values(dic_genre_word_count)
    uni_dic_tfidf = tf_idf_calculations.get_TFIDF_values(uni_dic_tf, uni_dic_idf)

    # calculate tfidf scores for bigram
    bi_dic_tf = tf_idf_calculations.get_TF_values(dic_genre_pair_count)
    bi_dic_idf = tf_idf_calculations.get_IDF_values(dic_genre_pair_count)
    bi_dic_tfidf = tf_idf_calculations.get_TFIDF_values(bi_dic_tf, bi_dic_idf)

    # classify input
    classification_uni = tester.classify_unigram(uni_dic_tfidf, text, False)
    classification_bi = tester.classify_bigram(bi_dic_tfidf, text, False)

    # set lambdas
    best_lambda = lambdas1.set_lambda(uni_dic_tfidf, bi_dic_tfidf, "Validation Set")

main()
