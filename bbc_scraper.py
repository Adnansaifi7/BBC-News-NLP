import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from gensim import corpora, models
import string
import time
import schedule
import threading

nltk.download('stopwords')

app = Flask(__name__)

cached_articles = []

stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

HEADERS = {"User-Agent": "Mozilla/5.0"}


def get_article_links(base_url="https://www.bbc.com/news"):
    response = requests.get(base_url, headers=HEADERS)
    if response.status_code != 200:
        print("Failed to fetch BBC News homepage.")
        return []

    soup = BeautifulSoup(response.text, 'html.parser')
    links = []

    for a in soup.select('a[href^="/news"]'):
        href = a.get('href')
        if href and href.startswith('/news') and not href.endswith('live'):
            full_url = f"https://www.bbc.com{href}"
            links.append(full_url)

    return list(set(links))[:10]  # Limit to 10 for performance


def scrape_article(url):
    time.sleep(1)
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to load article: {url}")
        return None

    soup = BeautifulSoup(response.text, 'html.parser')

    title_tag = soup.find('h1')
    title = title_tag.get_text(strip=True) if title_tag else 'No Title'

    paragraphs = soup.find_all('p')
    content = '\n'.join(p.get_text(strip=True) for p in paragraphs)

    publish_date = None
    if soup.find('time'):
        publish_date = soup.find('time').get('datetime')
    elif soup.find('meta', {'property': 'article:published_time'}):
        publish_date = soup.find('meta', {'property': 'article:published_time'}).get('content')

    author = None
    author_meta = soup.find('meta', {'name': 'byl'})
    if author_meta:
        author = author_meta.get('content')
    else:
        author_tag = soup.find(class_='byline__name')
        if author_tag:
            author = author_tag.get_text(strip=True)

    category = url.split('/')[4] if len(url.split('/')) > 4 else 'general'

    sentiment = analyze_sentiment(content)

    return {
        "title": title,
        "content": content,
        "url": url,
        "publish_date": publish_date,
        "author": author,
        "category": category,
        "sentiment": sentiment
    }


def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "Positive"
    elif polarity < -0.1:
        return "Negative"
    else:
        return "Neutral"


def preprocess_text(text):
    tokens = text.lower().split()
    tokens = [t.strip() for t in tokens if t not in stop_words and t not in punctuations and len(t) > 2]
    return tokens


def get_topics(documents, num_topics=3):
    texts = [preprocess_text(doc) for doc in documents]
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15)

    topics = lda_model.print_topics(num_words=4)
    return topics


@app.route("/")
def home():
    search_query = request.args.get('q', '').lower()
    category_filter = request.args.get('category', '').lower()
    page = request.args.get('page', 1, type=int)

    articles = cached_articles.copy()

    if search_query:
        articles = [a for a in articles if search_query in a['title'].lower() or search_query in a['content'].lower()]
    if category_filter:
        articles = [a for a in articles if category_filter == a['category'].lower()]

    ARTICLES_PER_PAGE = 3
    total_articles = len(articles)
    start = (page - 1) * ARTICLES_PER_PAGE
    end = start + ARTICLES_PER_PAGE
    paginated_articles = articles[start:end]

    has_next = end < total_articles
    contents = [a['content'] for a in paginated_articles]
    topics = get_topics(contents) if contents else []

    return render_template("articles.html", articles=paginated_articles, topics=topics,
                           search_query=search_query, category_filter=category_filter,
                           page=page, has_next=has_next)


def scrape_and_cache():
    global cached_articles
    print("Scraping articles and updating cache...")
    links = get_article_links()
    articles = []
    for link in links:
        article = scrape_article(link)
        if article:
            articles.append(article)
    cached_articles = articles
    print(f"Cached {len(articles)} articles.")


def run_schedule():
    schedule.every(1).hour.do(scrape_and_cache)
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    scrape_and_cache()
    scheduler_thread = threading.Thread(target=run_schedule, daemon=True)
    scheduler_thread.start()
    app.run(debug=True)


