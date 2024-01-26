import requests
import os
import re
import urllib.parse
from bs4 import BeautifulSoup

#assumption for first website: there are some quotes by Michael that contain comments from the narrator. I am assuming that this is too infrequent to make a difference, because distinguishing between the narrator and michael would be difficult.
#assumption for second website: any time a capital letter followed by a colon [A-Z]:, or 3 capital letters [A-Z]{3,} appear, it is considered to be someone other than Michael talking.
#task1 scrapes a website for a movie script, and only includes the quotes from a specific character
class Task1:
    def scrape(self, html1, html2):

        #get directory that contains script, create new text files in directory
        script_directory = os.path.dirname(os.path.realpath(__file__))
        relativeFilePath1 = os.path.join(script_directory, 'part2.txt')
        relativeFilePath2 = os.path.join(script_directory, 'part1.txt')

        #first html
        html1r = requests.get(html1)
        script1 = BeautifulSoup(html1r.text, features="html.parser") 
        with open(relativeFilePath1, 'w', encoding='utf-8') as file:
            michaelTags = script1.find_all('b', string=lambda text: text and re.match(r'^\t+MICHAEL', text)) #find all of Michael's quotes
            #for each Michael quote
            for michaelTag in michaelTags:
                quote = michaelTag.next_sibling #the sibling is the actual quote
                if quote:
                    formattedText = ' ' + re.sub(r'\s{2,}', ' ', quote.get_text(strip=True)) + '\n' #format the quote
                    file.write(formattedText)

        #second html
        url = html2
        with open(relativeFilePath2, 'w', encoding='utf-8') as file:
            #while there is a url for a next page (because this website has multiple pages for the script)
            while url:
                html2r = requests.get(url)
                script2 = BeautifulSoup(html2r.text, features="html.parser")
                michaelTags = script2.find_all('strong', string='MICHAEL:') #find all of Michael's quotes

                #for each Michael quote
                for michaelTag in michaelTags:
                    parent = michaelTag.find_parent('p') #the siblings of the parent of an instance of 'MICHAEL:' will contain the quote that michael says, so get the parent
                    siblings = [] #the website splits the lines of text strangely, so each sibling of the parent is appended to a list to make it into an actual full quote
                    nextSibling = parent.find_next_sibling('p')

                    #iterate over the siblings and append their text to 'siblings' list to build the Michael quote
                    while nextSibling:
                        if re.search(r'[A-Z]{3,}|[A-Z]:', nextSibling.text): #refer to assumption note above class Task1
                            break
                        siblings.append(nextSibling.text.strip())
                        nextSibling = nextSibling.find_next_sibling('p')
                    file.write(' '.join(siblings) + '\n') #write the quote to the text file

                nextPageLink = script2.find('a', string='Next\xa0»') #find the next page
                
                #if there is another page, go to it
                if nextPageLink:
                    relativeUrl = nextPageLink['href']
                    url = urllib.parse.urljoin(url, relativeUrl)

                #if not, break
                else:
                    break
                

class Main:
    executor = Task1()
    executor.scrape("https://imsdb.com/scripts/Godfather-Part-II.html", "https://www.scripts.com/script.php?id=the_godfather_71")
