import requests
from bs4 import BeautifulSoup

def extract_content(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.title.string if soup.title else "No Title"
    body = " ".join([p.get_text() for p in soup.find_all("p")])
    return {"url": url, "title": title, "body": body}
