import requests
import re
from bs4 import BeautifulSoup

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

class Main:
    executor = Task1()
    executor.scrape("godfather.txt1", "https://imsdb.com/scripts/Godfather.html")
    executor.scrape("godfather.txt2", "https://imsdb.com/scripts/Godfather-Part-II.html")
