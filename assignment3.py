import requests
from bs4 import BeautifulSoup
import os
import csv
import re
import nltk
from textblob import TextBlob
from nltk.corpus import cmudict, stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import gensim.downloader as api
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
    """
    A class for processing song data for training and testing. 
    """
    def processData(self, filepath):
        """
        Processes the data from a specified file path for training the neural network.

        Args:
            filepath (str): The path to the file containing the data.

        Returns:
            Tuple: A tuple containing the processed input data and labels.
        """
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
    
    def processTestData(self, folder_name):
        """
        Processes the test data from a specified file path.

        Args:
            filepath (str): The path to the file containing the test data.

        Returns:
            Tuple: A tuple containing the processed test input data and labels.
        """
        program_dir = os.path.dirname(os.path.realpath(__file__))  # Get the program's directory
        test_folder_path = os.path.join(program_dir, folder_name)  # Construct the full path to the test folder
        nn_input = []
        genre_list = []
        for genre in os.listdir(test_folder_path):
            if genre != '.DS_Store':
                genre_path = os.path.join(test_folder_path, genre)
                for song_file in os.listdir(genre_path):
                    song_path = os.path.join(genre_path, song_file)
                    with open(song_path, 'r', encoding='utf-8') as file:
                        song_text = file.read()
                    lyrics = task2.preProcess(song_text)
                    features = task2.extractFeatures(lyrics)
                    nn_input.append(features)
                    genre_list.append(genre.lower())

        return nn_input, genre_list


    @staticmethod
    def preProcess(text):
        """
        Preprocesses the text data.

        Args:
            text (str): The text to be preprocessed.

        Returns:
            str: The preprocessed text.
        """
        text = text.strip()
        text= text.lower()
        text = re.sub(r'[^\w\s\n\']', '', text)

        return text
    
    @staticmethod
    def genreIndex(genre):
        """
        Gets the index of a genre in the genres list.

        Args:
            genre (str): The genre name.

        Returns:
            int: The index of the genre in the genres list.
        """
        genres = ['rap', 'metal', 'rock', 'pop', 'country', 'blues']

        return genres.index(genre)
    
    @staticmethod
    def rhyme(word1, word2):
        """
        Checks if two words rhyme.

        Args:
            word1 (str): The first word.
            word2 (str): The second word.

        Returns:
            bool: True if the words rhyme, False otherwise.
        """
        return word1 in d and word2 in d and d[word1][0][-1:] == d[word2][0][-1:]

    @staticmethod
    def extractFeatures(lyrics):
        """
        Extracts features from the lyrics.

        Args:
            lyrics (str): The lyrics of the song.

        Returns:
            List: A list of features extracted from the lyrics.
        """
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

        # Get top 3 word frequency total
        stop_words = set(stopwords.words('english'))
        tokens_without_stopwords = [word for word in tokens if word.lower() not in stop_words]
        word_freq = Counter(tokens_without_stopwords)
        first3 = 0
        for word, freq in word_freq.most_common(3):
            first3 += freq

        # Get possible slang word count
        slang = 0
        for token in tokens:
            if token == "'":
                slang += 1

        # Get sentence length
        sentences = 0
        total = 0
        for sentence in lyrics.split('\n'):
            total += len(sentence.split())
            sentences += 1
        average_sentence_length = total/sentences

        features.append(song_length)
        features.append(sentiment.polarity)
        features.append(sentiment.subjectivity)
        features.append(rhyme_count)
        features.append(first3)
        features.append(slang)
        features.append(average_sentence_length)

        return features

class task3:
    @staticmethod
    def processData(filepath, word2vec_model):
        """
        Processes the data from a specified file path for training the neural network.

        Args:
            filepath (str): The path to the file containing the data.
            word2vec_model: Pre-trained Word2Vec model.

        Returns:
            Tuple: A tuple containing the processed input data and labels.
        """
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
                lyric_embedding = task3.getLyricEmbedding(lyrics, word2vec_model)
                nn_input.append(lyric_embedding)

            return nn_input, nn_labels
        
    @staticmethod
    def getLyricEmbedding(lyrics, word2vec_model):
        """
        Get the vector representation of the entire lyric by averaging word embeddings.

        Args:
            lyrics (str): The lyrics of the song.
            word2vec_model: Pre-trained Word2Vec model.

        Returns:
            numpy array: The vector representation of the lyric.
        """
        # Tokenize the lyrics into words
        tokens = word_tokenize(lyrics)

        # Filter out stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        # Get word embeddings for each word and average them
        embeddings = [word2vec_model[word] for word in tokens if word in word2vec_model]
        if embeddings:
            lyric_embedding = np.mean(embeddings, axis=0)
        else:
            # If no word in the lyric is found in the word2vec model, return zeros
            lyric_embedding = np.zeros(word2vec_model.vector_size)

        return lyric_embedding
    
    def processTestData(self, folder_name, word2vec_model):
        """
        Processes the test data from a specified file path.

        Args:
            filepath (str): The path to the file containing the test data.

        Returns:
            Tuple: A tuple containing the processed test input data and labels.
        """
        program_dir = os.path.dirname(os.path.realpath(__file__))  # Get the program's directory
        test_folder_path = os.path.join(program_dir, folder_name)  # Construct the full path to the test folder
        nn_input = []
        genre_list = []
        for genre in os.listdir(test_folder_path):
            if genre != '.DS_Store':
                genre_path = os.path.join(test_folder_path, genre)
                for song_file in os.listdir(genre_path):
                    song_path = os.path.join(genre_path, song_file)
                    with open(song_path, 'r', encoding='utf-8') as file:
                        song_text = file.read()
                    lyrics = task2.preProcess(song_text)
                    genre_list.append(genre.lower())
                    lyric_embedding = task3.getLyricEmbedding(lyrics, word2vec_model)
                    nn_input.append(lyric_embedding)

        return nn_input, genre_list


class songGenreClassifier(nn.Module):
    """
    A neural network classifier for predicting song genres based on song features.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the classifier with input, hidden, and output sizes.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The size of the hidden layer.
            output_size (int): The size of the output (number of genres).
        """
        super(songGenreClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        Performs forward pass through the neural network.

        Args:
            x (tensor): The input tensor.

        Returns:
            tensor: The output tensor after passing through the network.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def trainModel(self, train_input, train_labels, valid_input, valid_labels, num_epochs):
        """
        Trains the neural network model.

        Args:
            train_input (tensor): The input data for training.
            train_labels (tensor): The labels for training.
            valid_input (tensor): The input data for validation.
            valid_labels (tensor): The labels for validation.

        Returns:
            None
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=0.001)

        train_losses = []
        valid_losses = []

        # Training loop
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
        """
        Gets the predicted genre based on the highest probability.

        Args:
            predicted_probabilities (tensor): The predicted probabilities for each genre.

        Returns:
            str: The predicted genre.
        """
        genres = ['rap', 'metal', 'rock', 'pop', 'country', 'blues']
        predicted_genre_index = torch.argmax(predicted_probabilities, dim=0).item()
        predicted_genre = genres[predicted_genre_index]

        return predicted_genre
    
    @staticmethod
    def normalize_data(data, epsilon=1e-8):
        """
        Normalizes the input data.

        Args:
            data (tensor): The input data to be normalized.
            epsilon (float): Small value to prevent division by zero.

        Returns:
            tensor: The normalized data.
        """
        mean = torch.mean(data, axis=0)
        std = torch.std(data, axis=0)
        normalized_data = (data - mean) / (std + epsilon)

        return normalized_data


def main():
    word2vec_model = api.load("word2vec-google-news-300")

    # Get instances of task objects
    first_task = task1()
    second_task = task2()
    third_task = task3()

    #testsongspath = "/Users/evankoenig/Downloads/Test Songs"
    #first_task.scrapesongs(testsongspath)
    
    # Load, process, and format data
    train_input, train_labels = second_task.processData("trainingdata.csv")
    valid_input, valid_labels = second_task.processData("validationdata.csv")
    train_input = torch.tensor(train_input)
    train_labels = torch.tensor(train_labels)
    valid_input = torch.tensor(valid_input)
    valid_labels = torch.tensor(valid_labels)
    normalized_train_input = songGenreClassifier.normalize_data(train_input)
    normalized_valid_input = songGenreClassifier.normalize_data(valid_input)

    train_input2, train_labels2 = third_task.processData("trainingdata.csv", word2vec_model)
    valid_input2, valid_labels2 = third_task.processData("validationdata.csv", word2vec_model)
    train_input2 = torch.tensor(train_input2)
    train_labels2 = torch.tensor(train_labels2)
    valid_input2 = torch.tensor(valid_input2)
    valid_labels2 = torch.tensor(valid_labels2)
    normalized_train_input2 = songGenreClassifier.normalize_data(train_input2)
    normalized_valid_input2 = songGenreClassifier.normalize_data(valid_input2)

    # Instantiate/train model
    model = songGenreClassifier(7, 512, 6)
    model.trainModel(normalized_train_input, train_labels, normalized_valid_input, valid_labels, 45)

    model2 = songGenreClassifier(word2vec_model.vector_size, 512, 6)
    model2.trainModel(normalized_train_input2, train_labels2, normalized_valid_input2, valid_labels2, 45)

    # Test model
    test_input, genre_list = second_task.processTestData("Test Songs")
    test_input = torch.tensor(test_input)
    normalized_test_input = songGenreClassifier.normalize_data(test_input)
    with torch.no_grad():
        output = model(normalized_test_input)
        prediction_list = []
        for i in range(len(output)):
            predicted_probabilities = output[i]
            predicted_genre = model.getGenre(predicted_probabilities)
            prediction_list.append(predicted_genre)

        correctpredictions = 0
        for i in range(len(genre_list)):
            if prediction_list[i] == genre_list[i]:
                correctpredictions += 1

        print(genre_list)
        print(correctpredictions)
        print(prediction_list)
        
    test_input2, genre_list2 = third_task.processTestData("Test Songs", word2vec_model)
    test_input2 = torch.tensor(test_input2)
    normalized_test_input2 = songGenreClassifier.normalize_data(test_input2)
    with torch.no_grad():
        output = model2(normalized_test_input2)
        prediction_list = []
        for i in range(len(output)):
            predicted_probabilities = output[i]
            predicted_genre = model2.getGenre(predicted_probabilities)
            prediction_list.append(predicted_genre)

        correctpredictions = 0
        for i in range(len(genre_list2)):
            if prediction_list[i] == genre_list2[i]:
                correctpredictions += 1

        print(genre_list2)
        print(correctpredictions)
        print(prediction_list)    


if __name__ == "__main__":
    main()
