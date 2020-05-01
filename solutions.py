import re
import spacy
import seaborn as sns
import pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB


#######################################
# Exercise #1
#######################################

stemmer = SnowballStemmer("english")
not_alphanumeric_or_space = re.compile('[^(\w|\s|\d)]')
nlp = spacy.load('en_core_web_sm')

def preprocess(doc):
    doc = re.sub(not_alphanumeric_or_space, '', doc)
    words = [t.lemma_ for t in nlp(doc) if t.lemma_ != '-PRON-']
    return ' '.join(words).lower()

vectorizer = TfidfVectorizer(min_df=2,
                             max_df=.8,
                             preprocessor=preprocess,
                             stop_words='english',
                             use_idf=False,
                             norm=False)

v = vectorizer.fit_transform(docs)
v = np.asarray(v.todense())


# Get out the labels (keys of the vocabulary), sorted by index.
labels, _ = zip(*sorted(vectorizer.vocabulary_.items(), key=lambda x: x[1]))

# It might be reasonable to visualize the mean by
# plotting the dimensions as a bar plot:
sns.barplot(list(labels), v.mean(axis=0))

def plot_cov(d, labels, corr=False):
    d = pd.DataFrame(d, index=labels, columns=labels)
    sns.heatmap(d)

# Plot the covariance/correlation as a heatmap:
plot_cov(np.corrcoef(v, rowvar=False), labels, corr=True)


#######################################
# Exercise #2
#######################################

def get_class(vecs, y, target):
    idx = np.argwhere(y == target).flatten()
    return vecs[idx]

def get_multi(vecs, y, target):
    vecs = get_class(vecs, y, target)
    d = vecs.sum(axis=0) / vecs.sum()
    return d

# Plot the 0 class:
sns.barplot(list(labels), get_multi(v, y, 0))

# Plot the 1 class:
sns.barplot(list(labels), get_multi(v, y, 1))


# BONUS: Plot the difference:
def get_separation(vecs, y, labels, alpha=.1):
    # add smoothing value of alpha:
    vecs = alpha + vecs

    dif = np.log(get_multi(vecs, y, 1)) - np.log(get_multi(vecs, y, 0))
    # print(np.log(get_multi(vecs, y, 1)))
    # print(get_multi(vecs, y, 1)))
    idxs = np.argsort(np.abs(dif))

    sorted_labels = [labels[i] for i in idxs]
    sorted_difs = [dif[i] for i in idxs]

    return sorted_labels, sorted_difs


sns.barplot(*get_separation(v, y, labels))

#######################################
# Exercise #3
#######################################

def conditional_cov(v, y, target, corr=False):
    # Note that some rows/cols will have
    # perfect conditional correlation because they
    # never show up in a given class (always 0!)

    # L1 norm is the same as turning freq into probability:
    vv = v / np.linalg.norm(v, ord=1, axis=1).reshape(-1, 1)

    # t = t_i - E[t_i | Y]
    t = get_class(vv, y, target) - get_multi(vv, y, target)

    if corr:
        return np.corrcoef(t, rowvar=False)

    # E[t*t | Y]
    return t.T.dot(t) / vv.shape[0]



plot_cov(conditional_cov(v, y, 0), labels)

plot_cov(conditional_cov(v, y, 1), labels)

# We note that certain words have positive correlations
# that go beyond the class. For example:
#
# milk <> bone, strong
# milk <> mother
#
# What does this say about the words milk and bone?
#
# And:
#
# tiger <> elephant
#
# What does this say about tigers and elephants?
#
# And in class 0:
#
# ghost <> mother, tell
#


#######################################
# Exercise #4
#######################################

yelps = yelps.sample(frac=1.)
V = vectorizer.fit_transform(yelps.text)
y = yelps.positive

def _cv_score(V, y, models):

    # list of dictionaries with name of model and mean cv score
    return [{ 'name': name, 'value': np.mean(cross_val_score(model, V, y, cv=3)) }
            for name, model in models]

def plot_cv_scores(V, y, models, ticks):

    # get cv scores for each model at each data level
    scores = [_cv_score(V[:t], y[:t], models) for t in ticks]

    # Plot the X axis on log scale so it's easier to see changes
    # with log N observations:
    scores = [({'N': np.log(t), **i}) for t,s
              in zip(ticks, scores) for i in s]

    df = pd.DataFrame(scores)
    return sns.lineplot(y='value', x='N', hue='name', data=df)

models = [('NB', MultinomialNB(fit_prior=False)),
          ('LR', LogisticRegression())]

plot_cv_scores(V, y, models, [300, 600, 1200, 2400, 4800, 9600])
