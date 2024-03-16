import requests
from bs4 import BeautifulSoup
import os
import csv
import re
import nltk
from textblob import TextBlob
from nltk.corpus import cmudict
from nltk.tokenize import word_tokenize
from collections import Counter
nltk.download('cmudict')

# CMU pronouncing dictionary for rhyme count
d = cmudict.dict()

class task1:
    def scrapesongs(path):
        with open('trainingdata.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Song Name', 'Genre', 'Lyrics'])
            for genre in os.listdir(path):
                if genre != '.DS_Store':
                    if genre == 'Metal':
                        html = requests.get('https://www.popvortex.com/music/charts/top-heavy-metal-songs.php')
                    else:
                        html = requests.get(f'https://www.popvortex.com/music/charts/top-{genre.lower()}-songs.php')
                    soup = BeautifulSoup(html.text, features='html.parser')
                    song_spans = soup.find_all('cite', class_='title')
                    artist_spans = soup.find_all('em',class_='artist')
                    for i in range(len(song_spans)):
                        song_name = re.sub(r'[^a-zA-Z]', '-', song_spans[i].text)
                        artist_name = re.sub(r'[^a-zA-Z]', '-', artist_spans[i].text)
                        lyricshtml = requests.get(f'https://genius.com/{artist_name}-{song_name}-lyrics')
                        lyricsoup = BeautifulSoup(lyricshtml.text, features='html.parser')
                        lyrics = lyricsoup.find('div', class_='Lyrics__Container-sc-1ynbvzw-1 kUgSbL')
                        if song_name not in path:
                            if lyrics:
                                cleanedlyrics = lyrics.get_text(separator='. ')
                                pattern = r'\[.*?\]'
                                cleanedlyrics = re.sub(pattern, '', cleanedlyrics)
                                csv_writer.writerow([song_name, artist_name,genre,cleanedlyrics])

class task2:

    def processData(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            file.readline()
            nn_input = []
            for row in reader:
                genre = row[2]
                lyrics = row[3]
                genre = task2.preProcess(genre)
                lyrics = task2.preProcess(lyrics)
                features = task2.extractFeatures(lyrics)
                nn_input.append(features) # not finalized, needs to be a vector and also will need labels

    @staticmethod
    def preProcess(text):
        text = text.strip()
        text= text.lower()
        text = re.sub(r'[^\w\s.]', '', text)

        return text
    
    @staticmethod
    def rhyme(word1, word2):
        return word1 in d and word2 in d and d[word1][0][-1:] == d[word2][0][-1:]

    @staticmethod
    def extractFeatures(lyrics):
        features = []

        # Get song length
        tokens = word_tokenize(lyrics)
        tokens.pop(0) # Remove leading period
        song_length = len(tokens)

        # Get text sentiment
        sentiment = TextBlob(lyrics).sentiment

        # Get rhyme count
        rhyme_count = 0
        last_words = []
        for i in range(len(tokens)):
            if tokens[i] == '.' and tokens[i-1] != '.':
                last_words.append(tokens[i-1])

        for i in range(len(last_words) - 1):
            if task2.rhyme(last_words[i], last_words[i+1]):
                rhyme_count += 1


        print("Song length:", song_length)
        print("Sentiment:", sentiment)
        print("Rhyme count", rhyme_count)
        features.append(song_length)
        features.append(sentiment)
        features.append(rhyme_count)

        return features

class task3:
    pass

def main():
    # Get instances of task objects
    first_task = task1()
    second_task = task2()
    third_task = task3()

    testsongspath = "/Users/evankoenig/Downloads/Test Songs"
    first_task.scrapesongs(testsongspath)

if __name__ == "__main__":
    main()

