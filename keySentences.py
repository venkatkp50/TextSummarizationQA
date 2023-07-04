import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.cluster.util import cosine_distance
import numpy as np
import re
import streamlit as st

MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
damping = 0.85  # damping coefficient, usually is .85
min_diff = 1e-5  # convergence threshold
steps = 100  # iteration steps
text_str = None
sentences = None
pr_vector = None



def prepData(data):
    tokenized_sent = sent_tokenize(data)
    tokenized_word = [word_tokenize(sent) for sent in tokenized_sent]

def normalize_whitespace(text):
      return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)

def _replace_whitespace(match):
    text = match.group()

    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "
    
def is_blank(string):
    return not string or string.isspace()

def get_symmetric_matrix(matrix):
    return matrix + matrix.T - np.diag(matrix.diagonal())

def core_cosine_similarity(vector1, vector2):

    return 1 - cosine_distance(vector1, vector2)


def _sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []

    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]


    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for w in sent1:
        if w in stopwords:
            continue
        
        vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

    return core_cosine_similarity(vector1, vector2)

def _build_similarity_matrix(sentences, stopwords=None):
    sm = np.zeros([len(sentences), len(sentences)])
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            sm[idx1][idx2] = _sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)
              
        min_diff
        
    sm = get_symmetric_matrix(sm)
    norm = np.sum(sm, axis=0)
    sm_norm = np.divide(sm, norm, where=norm != 0)  # this is ignore the 0 element in norm

    return sm_norm

def _run_page_rank(similarity_matrix):

    pr_vector = np.array([1] * len(similarity_matrix))

    previous_pr = 0
    for epoch in range(steps):
        pr_vector = (1 - damping) + damping * np.matmul(similarity_matrix, pr_vector)
        if abs(previous_pr - sum(pr_vector)) < min_diff:
            break
        else:
            previous_pr = sum(pr_vector)

    return pr_vector

def _get_sentence( index):
    try:
        return sentences[index]
    except IndexError:
        return ""


def get_top_sentences( number,pr_vector):

    top_sentences = []

    if pr_vector is not None:
        sorted_pr = np.argsort(pr_vector)
        sorted_pr = list(sorted_pr)
        sorted_pr.reverse()

        index = 0
        for epoch in range(number):
            sent = sentences[sorted_pr[index]]
            sent = normalize_whitespace(sent)
            top_sentences.append(sent)
            index += 1

    return top_sentences

def analyze(text, stop_words=None):
    text_str = text
    sentences = sent_tokenize(text_str)
    tokenized_sentences = [word_tokenize(sent) for sent in sentences]
    similarity_matrix = _build_similarity_matrix(tokenized_sentences, stop_words)
    pr_vector = _run_page_rank(similarity_matrix)
    #st.write(pr_vector)

    top_sentences = []

    if pr_vector is not None:
        sorted_pr = np.argsort(pr_vector)
        sorted_pr = list(sorted_pr)
        sorted_pr.reverse()
        
        #st.write(sorted_pr)

        index = 0
        for epoch in range(3):
            sent = sentences[sorted_pr[index]]
            sent = normalize_whitespace(sent)
            top_sentences.append(sent)
            index += 1

    return top_sentences
    #return get_top_sentences(5,pr_vector)







# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# import numpy as np
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import streamlit as st
# from GPT2Summarizer import getGPT2Summary

# GPT2_MAX_TOKEN = 1024

# def multiSummarizer(filenames, n):
#     docs = []
#     intro = []
#     for filen in filenames:
#         with open('text_file/'+filen, "r") as fd:
#             multifile_text = fd.readlines()
#             header =[]
#             para = []
#             for line in multifile_text:
#                 if len(line) > 1:
#                     if len(line) < 100:
#                         header.append(line)
#                     else:
#                         para.append(line)  

#             # st.write(header[0])
#             # intro.append(para[0])
#             # multi_fulltext = ''.join( lines for lines in para)
#             # #docs.append(multi_fulltext)
#             multi_fulltext = ''.join(lines for lines in para)   
#             docs.append(multi_fulltext)


#     #st.write(getGPT2Summary(1,'GPT2',introduction,GPT2_MAX_TOKEN,100))
    
#     st.write('docs=',len(docs),'.......................')
#     vectorizer = TfidfVectorizer(stop_words='english')
#     X = vectorizer.fit_transform(docs)
#     similarity_matrix = cosine_similarity(X)
#     scores = np.ones(len(docs))
#     damping_factor = 0.85
#     tol = 1e-5
#     max_iter = 100
#     for i in range(max_iter):
#         prev_scores = scores.copy()
#         for j in range(len(docs)):
#             scores[j] = (1 - damping_factor) + damping_factor * \
#                         np.sum(similarity_matrix[j] * scores / np.sum(similarity_matrix[j]))
#         if np.linalg.norm(scores - prev_scores) < tol:
#             break
                    
#     sentence_scores = list(zip(range(len(docs)), scores))
#     sentence_scores.sort(key=lambda x: x[1], reverse=True)
#     selected_indices = [x[0] for x in sentence_scores[:n]]
#     selected_indices.sort()
#     summary = ' '.join([nltk.sent_tokenize(docs[i])[0] for i in selected_indices])
#     return summary
