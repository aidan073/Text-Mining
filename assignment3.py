import requests
from bs4 import BeautifulSoup
import os
import csv
import re
import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from collections import Counter

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

    def getData(self, filepath):
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            file.readline()
            for row in reader:
                genre = row[2]
                lyrics = row[3]
                genre = task2.preProcess(genre)
                lyrics = task2.preProcess(lyrics)
                task2.extractFeatures(lyrics)

    @staticmethod
    def preProcess(text):
        text = text.strip()
        text= text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        return text

    @staticmethod
    def extractFeatures(lyrics):
        words = word_tokenize(lyrics)
        word_freq = Counter(words)
        sentiment = TextBlob(lyrics).sentiment
        print("Word frequency:", word_freq)
        print("Sentiment:", sentiment)

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

