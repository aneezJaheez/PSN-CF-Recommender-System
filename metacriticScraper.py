#the following program is a simple scraping algorithm that collects username and their rating information from the website metacritic.com. 
#the code below only deals with ps4 games have a minimum rating of 1

from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup
from urllib.request import Request
from urllib.error import HTTPError

def getPageSoup(page_url):
	#Sometimes GATEWAY error 504 occurs, which is a server side error. THis piece of code will prevent that from happening by repeating the request if it happens
	while True:
		try:
			preClient = Request(page_url, headers = {"User-Agent": "Mozilla/5.0"})
			client = uReq(preClient)
			web_html = client.read()
			web_soup = soup(web_html, "html.parser")
		except:
			continue

	client.close()

	return web_soup

#function to add the intial part of the website to the rettrieved url
def optUrl(web_url):
	return "https://www.metacritic.com" + web_url

filename = "metacritic.csv"
f = open(filename, "w")

headers = "Game_Name, User_Name, User_Rating\n"
f.write(headers)

#starting page from which data will be read and website will be traversed
my_url = "https://www.metacritic.com/browse/games/score/metascore/all/ps4/filtered?page=0"

while True:
	#retrieving the page soup that contains each game
	page_soup = getPageSoup(my_url)

	game_containers = page_soup.findAll("td", {"class" : "clamp-summary-wrap"})

	#looping through every game to open it and read the reviews
	for container in game_containers:
		name_href = container.findAll('a', {'class' : 'title'})[0]
		game_name = name_href.h3.text

		print("game name : " + game_name)

		#opening first game review page
		reviews_url = optUrl(name_href['href']) + "/user-reviews"
		rev_soup = getPageSoup(reviews_url)

		#this loop goes through every review page
		while True:
			#collecting all individual user rating data
			rev_containers = rev_soup.findAll("li", {"class" : ['review', 'user-review']})

			#this loop goes through every review in a single review page
			for r_container in rev_containers:
				user_score = r_container.find("div", {"class" : "review_grade"}).div.text
				try:
					user_name = r_container.find("div", {"class" : "name"}).a.text
				except:
					break

				print("username : " + user_name)
				print("userscore : " + user_score)

				# writing contents into text file
				f.write(game_name.replace(",", " ") + ',' + user_name + "," + user_score + "\n")

			#tries to retrive the next page of reviews. breaks out of loop if next page does not exist
			try:
				nextrev_container = rev_soup.find("span", {"class" : "flipper next"}).a['href']
				nextrev_url = optUrl(nextrev_container)
			except:
				break

			#retrieving the next review page for next iteration
			reviews_url = nextrev_url
			rev_soup = getPageSoup(reviews_url)

	#tries to retrieve next page of games. Breaks out of loop if next page does not exist.
	try:
		nextbtn_container = page_soup.find("span", {"class" : "flipper next"}).a['href']
		my_url = optUrl(nextbtn_container)
	except:
		break

#end of file write process
f.close()

#end of scrape
