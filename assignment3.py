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
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
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
            nn_labels = []
            nn_input = []
            file.readline()

            for row in reader:
                genre = row[2]
                genre = task2.preProcess(genre)
                nn_labels.append(task2.genreIndex(genre))

                lyrics = row[3]
                lyrics = task2.preProcess(lyrics)
                features = task2.extractFeatures(lyrics)
                nn_input.append(features)

            return nn_input, nn_labels
    
    def processTestData(self, filepath):
        nn_input = []
        for genre in os.listdir(filepath):
            genre_path = os.path.join(filepath, genre)
            for song_file in os.listdir(genre_path):
                song_path = os.path.join(genre_path, song_file)
                with open(song_path, 'r', encoding='utf-8') as file:
                    song_text = file.read()
                lyrics = task2.preProcess(song_text)
                features = task2.extractFeatures(lyrics)
                nn_input.append(features)

        return nn_input


    @staticmethod
    def preProcess(text):
        text = text.strip()
        text= text.lower()
        text = re.sub(r'[^\w\s\n]', '', text)

        return text
    
    @staticmethod
    def genreIndex(genre):
        genres = ['rap', 'metal', 'rock', 'pop', 'country', 'blues']

        return genres.index(genre)
    
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
            if tokens[i] == '\n' and tokens[i-1] != '\n':
                last_words.append(tokens[i-1])

        for i in range(len(last_words) - 1):
            if task2.rhyme(last_words[i], last_words[i+1]):
                rhyme_count += 1

        features.append(song_length)
        features.append(sentiment.polarity)
        features.append(sentiment.subjectivity)
        features.append(rhyme_count)

        return features

class task3:
    pass

class songGenreClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(songGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def trainModel(self, train_input, train_labels, valid_input, valid_labels):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        train_losses = []
        valid_losses = []

        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            # Training
            self.train()
            optimizer.zero_grad()
            train_outputs = self(train_input)
            train_loss = criterion(train_outputs, train_labels)
            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

            # Validation
            self.eval()
            with torch.no_grad():
                valid_outputs = self(valid_input)
                valid_loss = criterion(valid_outputs, valid_labels)
                valid_losses.append(valid_loss.item())

            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item()}, Valid Loss: {valid_loss.item()}")

        # Plotting
        plt.plot(range(1, num_epochs+1), train_losses, label='Train Loss')
        plt.plot(range(1, num_epochs+1), valid_losses, label='Valid Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def getGenre(self, predicted_probabilities):
        # Get the genre with the highest probability
        genres = ['rap', 'metal', 'rock', 'pop', 'country', 'blues']
        predicted_genre_index = torch.argmax(predicted_probabilities, dim=0).item()
        predicted_genre = genres[predicted_genre_index]

        return predicted_genre
    
    @staticmethod
    def normalize_data(data, epsilon=1e-8):
        mean = torch.mean(data, axis=0)
        std = torch.std(data, axis=0)
        normalized_data = (data - mean) / (std + epsilon)

        return normalized_data


def main():
    # Get instances of task objects
    first_task = task1()
    second_task = task2()
    third_task = task3()

    #testsongspath = "/Users/evankoenig/Downloads/Test Songs"
    #first_task.scrapesongs(testsongspath)
    
    # Load and process data
    train_input, train_labels = second_task.processData("trainingdata.csv")
    valid_input, valid_labels = second_task.processData("validationdata.csv")

    train_input = torch.tensor(train_input)
    train_labels = torch.tensor(train_labels)
    valid_input = torch.tensor(valid_input)
    valid_labels = torch.tensor(valid_labels)
    normalized_train_input = songGenreClassifier.normalize_data(train_input)
    normalized_valid_input = songGenreClassifier.normalize_data(valid_input)

    model = songGenreClassifier(4, 512, 6)
    model.trainModel(normalized_train_input, train_labels, normalized_valid_input, valid_labels)

    test_input = second_task.processTestData(r"C:\Users\Gigabyte\Desktop\Text Mining\assignment3\Test Songs")
    test_input = torch.tensor(test_input)
    normalized_test_input = songGenreClassifier.normalize_data(test_input)
    with torch.no_grad():
        output = model(normalized_test_input)
        for i in range(len(output)):
            predicted_probabilities = output[i]
            predicted_genre = model.getGenre(predicted_probabilities)
        


if __name__ == "__main__":
    main()
