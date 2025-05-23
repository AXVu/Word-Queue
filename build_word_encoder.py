from gensim.models import Word2Vec
import wikipedia
from nltk import word_tokenize, sent_tokenize
import re

def get_articles(search_terms, results, include_random=True):
    article_text = {}
    for term in search_terms:
        articles = wikipedia.search(term,results=results)
        for article in articles:
            try:
                article_text[article] = wikipedia.page(article, auto_suggest=False).content
            except wikipedia.exceptions.DisambiguationError as e:
                for a in e.options:
                    try:
                        article_text[a] = wikipedia.page(a, auto_suggest=False).content
                        print(f"accepted alt {a}")
                    except wikipedia.exceptions.DisambiguationError as e:
                        print(f"tossing alt {a}")
                    except wikipedia.exceptions.PageError as e:
                        print(f"tossing alt {a}")
                    except:
                        print("unknown error, tossing page")
            except wikipedia.exceptions.PageError as e:
                print(f"tossing {article} p")
                
    return article_text

def tokenize_article(article_text):
    article_text = article_text.lower()
    article_text = re.sub(r'[^a-zA-Z\s.]', '', article_text)
    sentences = sent_tokenize(article_text)
    tokenized_text = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_text

def build_corpus(search_terms, results, include_random=True):
    corpus = []
    articles = get_articles(search_terms, results, include_random=include_random)
    for title,content in articles.items():
        corpus.extend(tokenize_article(content))
    return corpus

corpus = build_corpus(["philosophy",
                       "government",
                       "cooking",
                       "writing",
                       "machine learning",
                       "zucchini",
                       "socks",
                       "autism",
                       "alcohol",
                       "principally",
                       "speech",
                       "communication",
                       "literature",
                       "mythology",
                       "history",
                       "cheese",
                       "medicine",
                       "math",
                       "statistics"], 50)
model = Word2Vec(sentences=corpus, vector_size=128, min_count=5, max_final_vocab=2048)
model.save("./vectorizers/initial_model_128f_2048vocab.model")