"""
mini_search_engine.py
A single-file, minimal search engine: crawler + indexer + web UI

Features:
- Crawl a small set of pages from a seed URL (same-domain, depth-limited)
- Extract plain text with BeautifulSoup
- Build an inverted index and TF-IDF vectors (pure Python, no heavy deps)
- Simple Flask web UI with a search box and results with snippets

Dependencies:
- Python 3.8+
- Flask
- requests
- beautifulsoup4

Install dependencies:
    pip install flask requests beautifulsoup4

Run:
    python mini_search_engine.py
Then open http://127.0.0.1:5000

Notes / Caveats:
- This is educational: not production-ready. No persistence beyond runtime.
- Be mindful of crawling: keep the crawl small (max_pages, same domain). Respect sites' robots.txt when using for real.

"""

from flask import Flask, request, render_template_string, redirect, url_for, jsonify
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import threading
import math
import re
from collections import deque, defaultdict, Counter

app = Flask(__name__)

# ---------- Simple crawler ----------

doc_texts = {}        # url -> full text

lock = threading.Lock()

# Basic tokenizer
_token_re = re.compile(r"\w+", re.UNICODE)

def tokenize(text):
    return [t.lower() for t in _token_re.findall(text)]


def fetch_text(url, timeout=8):
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "MiniSearchBot/1.0"})
        if 'text/html' not in resp.headers.get('Content-Type', ''):
            return None
        soup = BeautifulSoup(resp.text, 'html.parser')
        # Remove scripts/styles
        for s in soup(['script', 'style', 'noscript']):
            s.decompose()
        text = soup.get_text(separator=' ', strip=True)
        return text
    except Exception as e:
        print(f"fetch error {url}: {e}")
        return None


def crawl(seed_url, max_pages=50, max_depth=2):
    parsed_seed = urlparse(seed_url)
    base_netloc = parsed_seed.netloc

    q = deque()
    q.append((seed_url, 0))
    seen = set()

    while q and len(seen) < max_pages:
        url, depth = q.popleft()
        if url in seen or depth > max_depth:
            continue
        text = fetch_text(url)
        if not text:
            seen.add(url)
            continue
        with lock:
            doc_texts[url] = text
        seen.add(url)

        # parse links and add same-domain URLs
        try:
            soup = BeautifulSoup(text, 'html.parser')
        except Exception:
            soup = None
        if soup is None:
            continue
        for a in soup.find_all('a', href=True):
            href = a['href']
            abs_url = urljoin(url, href)
            p = urlparse(abs_url)
            # only same domain, ignore fragments
            if p.scheme not in ('http', 'https'):
                continue
            if p.netloc != base_netloc:
                continue
            clean = p._replace(fragment='').geturl()
            if clean not in seen:
                q.append((clean, depth+1))

    return len(doc_texts)

# ---------- Indexer: inverted index + tf-idf ----------

inverted_index = defaultdict(set)   # term -> set(url)
tf = {}     # url -> Counter(term -> count)
doc_count = 0
idf = {}    # term -> idf
norms = {}  # url -> vector norm for cosine

def build_index():
    global inverted_index, tf, doc_count, idf, norms
    inverted_index = defaultdict(set)
    tf = {}
    docs = list(doc_texts.items())
    doc_count = len(docs)

    for url, text in docs:
        tokens = tokenize(text)
        counts = Counter(tokens)
        tf[url] = counts
        for term in counts:
            inverted_index[term].add(url)

    # compute idf
    idf = {}
    for term, urls in inverted_index.items():
        df = len(urls)
        idf[term] = math.log((doc_count + 1) / (df + 1)) + 1.0

    # compute norms for each doc vector
    norms = {}
    for url, counts in tf.items():
        s = 0.0
        for term, cnt in counts.items():
            tf_val = 1 + math.log(cnt) if cnt > 0 else 0.0
            w = tf_val * idf.get(term, 0.0)
            s += w * w
        norms[url] = math.sqrt(s) if s > 0 else 1.0

# ---------- Search ----------

def query_to_vector(query):
    tokens = tokenize(query)
    q_counts = Counter(tokens)
    qvec = {}
    s = 0.0
    for term, cnt in q_counts.items():
        tf_q = 1 + math.log(cnt) if cnt > 0 else 0.0
        idf_t = idf.get(term, math.log((doc_count + 1) / 1) + 1.0)
        w = tf_q * idf_t
        qvec[term] = w
        s += w * w
    qnorm = math.sqrt(s) if s > 0 else 1.0
    return qvec, qnorm


def search(query, top_n=10):
    qvec, qnorm = query_to_vector(query)
    scores = []
    # candidate URLs: union of posting lists for query terms
    candidates = set()
    for term in qvec:
        candidates.update(inverted_index.get(term, set()))
    if not candidates:
        return []

    for url in candidates:
        score = 0.0
        counts = tf.get(url, {})
        for term, q_w in qvec.items():
            doc_cnt = counts.get(term, 0)
            if doc_cnt == 0:
                continue
            tf_d = 1 + math.log(doc_cnt)
            w_d = tf_d * idf.get(term, 0.0)
            score += q_w * w_d
        denom = (qnorm * norms.get(url, 1.0))
        score = score / denom if denom != 0 else 0.0
        scores.append((score, url))

    scores.sort(reverse=True)
    results = [(url, score) for score, url in scores[:top_n]]
    return results

# ---------- Snippet helper ----------

def make_snippet(url, query_terms, window=40):
    text = doc_texts.get(url, '')
    tokens = tokenize(text)
    # find first occurrence index of any query term
    qset = set(t.lower() for t in query_terms)
    idx = next((i for i, t in enumerate(tokens) if t in qset), 0)
    start = max(0, idx - window//2)
    end = min(len(tokens), start + window)
    snippet = ' '.join(tokens[start:end])
    return snippet + '...'

# ---------- Flask routes / UI ----------

INDEX_HTML = """
<!doctype html>
<title>Mini Search Engine</title>
<h1>Mini Search Engine</h1>
<form action="/do_search" method="get">
  <input name="q" placeholder="Search..." size=50 autofocus>
  <input type="submit" value="Search">
</form>
<p>Documents indexed: {{doc_count}}</p>
<hr>
<h3>Index / Crawl control</h3>
<form action="/crawl" method="post">
  Seed URL: <input name="seed" size=40 value="https://example.com">
  Max pages: <input name="max_pages" size=3 value="10">
  Max depth: <input name="max_depth" size=3 value="1">
  <input type="submit" value="Start Crawl">
</form>
<form action="/reindex" method="post" style="margin-top:8px;">
  <input type="submit" value="Rebuild Index">
</form>
<hr>
{% if results is defined %}
  <h2>Results for: {{query}}</h2>
  {% if results %}
    <ol>
    {% for url, score, snippet in results %}
      <li>
        <a href="{{url}}" target="_blank">{{url}}</a> <br>
        <small>score: {{"{:.4f}".format(score)}}</small>
        <p>{{snippet}}</p>
      </li>
    {% endfor %}
    </ol>
  {% else %}
    <p>No results.</p>
  {% endif %}
{% endif %}

"""

@app.route('/')
def index():
    return render_template_string(INDEX_HTML, doc_count=doc_count)

@app.route('/crawl', methods=['POST'])
def crawl_route():
    seed = request.form.get('seed')
    max_pages = int(request.form.get('max_pages', 10))
    max_depth = int(request.form.get('max_depth', 1))
    if not seed:
        return redirect(url_for('index'))

    def _job():
        try:
            n = crawl(seed, max_pages=max_pages, max_depth=max_depth)
            build_index()
            print(f"crawl finished, docs={n}")
        except Exception as e:
            print("crawl error:", e)

    t = threading.Thread(target=_job, daemon=True)
    t.start()
    return redirect(url_for('index'))

@app.route('/reindex', methods=['POST'])
def reindex_route():
    build_index()
    return redirect(url_for('index'))

@app.route('/do_search')
def do_search():
    q = request.args.get('q', '')
    if not q:
        return redirect(url_for('index'))
    results = search(q, top_n=20)
    enriched = []
    for url, score in results:
        snippet = make_snippet(url, q.split())
        enriched.append((url, score, snippet))
    return render_template_string(INDEX_HTML, doc_count=doc_count, results=enriched, query=q)

@app.route('/api/search')
def api_search():
    q = request.args.get('q', '')
    n = int(request.args.get('n', 10))
    if not q:
        return jsonify([])
    results = search(q, top_n=n)
    out = []
    for url, score in results:
        out.append({"url": url, "score": score, "snippet": make_snippet(url, q.split())})
    return jsonify(out)

if __name__ == '__main__':
    print("Mini search engine starting. Open http://127.0.0.1:5000")
    app.run(debug=True, port=5000)
