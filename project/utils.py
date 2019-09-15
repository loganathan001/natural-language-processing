import nltk
import pickle
import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    # Hint: you have already implemented a similar routine in the 3rd assignment.
    # Note that here you also need to know the dimension of the loaded embeddings.
    # When you load the embeddings, use numpy.float32 type as dtype

    starspace_embeddings = dict()
    for line in open(embeddings_path, encoding='utf-8'):
        row = line.strip().split('\t')
        starspace_embeddings[row[0]] = np.array(row[1:], dtype=np.float32)

    return starspace_embeddings, starspace_embeddings[next(iter(starspace_embeddings))].shape[0]


def question_to_vec(question, embeddings, dim=300):
    """
        question: a string
        embeddings: dict where the key is a word and a value is its' embedding
        dim: size of the representation

        result: vector representation for the question
    """
#     vector = np.array([ wv_embeddings[word] for word in question.split() if word in wv_embeddings ])
#     mean = np.zeros(dim)
#     if len(vector) != 0:
#         mean = vector.mean(axis=0)
     
#     return mean

    result = np.zeros(dim)
    cnt = 0
    words = question.split()
    for word in words:
        if word in embeddings:
            result += np.array(embeddings[word])
            cnt += 1
    if cnt != 0:
        result /= cnt
    return result


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def rank_candidates(question, candidates, embeddings, dim=300):
    """
        question: a string
        candidates: a list of strings (candidates) which we want to rank
        embeddings: some embeddings
        dim: dimension of the current embeddings
        
        result: a list of pairs (initial position in the list, question)
    """

    q_vecs = np.array([question_to_vec(question, embeddings, dim) for i in range(len(candidates))])
    cand_vecs = np.array([question_to_vec(candidate, embeddings, dim) for candidate in candidates])
    cosines = np.array(cosine_similarity(q_vecs, cand_vecs)[0])
    merged_list = list(zip(cosines, range(len(candidates)), candidates))

    sorted_list  = sorted(merged_list, key=lambda x: x[0], reverse=True)
    result = [(b,c) for a,b,c in sorted_list]
    return result
