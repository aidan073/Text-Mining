import requests
import re
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from wordcloud import WordCloud

#assumption: There are some quotes by Michael that contain comments from the narrator. I am assuming that this is too infrequent to make a difference, because distinguishing between the narrator and michael would be difficult.
#task1 scrapes a website for a movie script, and only includes the quotes from a specific character
class Task1:
    def scrape(self, output, link):
        #get link
        html = requests.get(link)
        script = BeautifulSoup(html.text, features="html.parser") 
        with open(output, 'w', encoding='utf-8') as file:
            michaelTags = script.find_all('b', string=lambda text: text and re.match(r'^\t+MICHAEL', text)) #find all of Michael's quotes
            #for each Michael quote
            for michaelTag in michaelTags:
                quote = michaelTag.next_sibling #the sibling is the actual quote
                if quote:
                    formattedText = ' ' + re.sub(r'\s{2,}', ' ', quote.get_text(strip=True)) + '\n' #format the quote
                    file.write(formattedText)

class Task2:
    def wordCloud(self, textFile1, textFile2):
        with open(textFile1, 'r') as file:
            contents1 = file.read()
        with open(textFile2, 'r') as file:
            contents2 = file.read()
        
        wordcloud1 = WordCloud(width=600, height=200, background_color='white').generate(contents1)
        wordcloud2 = WordCloud(width=600, height=200, background_color='white').generate(contents2)
        plt.figure(figsize=(5, 3))
        plt.imshow(wordcloud1)
        plt.axis('off')
        plt.show(block=False)
        plt.figure(figsize=(5, 3))
        plt.imshow(wordcloud2)
        plt.axis('off')
        plt.show()

class Main:
    executor1 = Task1()
    executor1.scrape("godfather1.txt", "https://imsdb.com/scripts/Godfather.html")
    executor1.scrape("godfather2.txt", "https://imsdb.com/scripts/Godfather-Part-II.html")

    executor2 = Task2()
    executor2.wordCloud("godfather1.txt", "godfather2.txt")
