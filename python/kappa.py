from __future__ import print_function
from __future__ import division
import os
import sys
import getopt
import numpy as np

'''
join -o 1.2,2.2 -t $'\t' <(sort test_scores.txt) <(sort test_z.txt) | python kappa.py -[i,f]
'''

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat


def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

''' expects rater_a to be int  '''
def quadratic_weighted_kappa(rater_a, rater_b):

    rater_a = np.round(rater_a).astype('int32')
    min_rating = min(rater_a)
    max_rating = max(rater_a)
    
    rater_b = np.clip(rater_b, min_rating, max_rating)
    rater_b = np.round(rater_b).astype('int32')
    
#     try:
#         rater_b = np.array(rater_b, dtype='int32')
#     except ValueError:
#         rater_b = np.array(rater_b, dtype='float32')
#         rater_b = np.clip(rater_b, min_rating, max_rating)
#         rater_b = np.round(rater_b).astype('int32')

    assert(len(rater_a) == len(rater_b))

#     min_rating = min(min_rating, min(rater_b))
#     max_rating = max(max_rating, max(rater_b))

    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

## int kappa
def ikappa(t, x, w=None):
    t = np.array(t, dtype='float32')
    x = np.array(x, dtype='float32')
    if w:
        t*=w
        x*=w
    return quadratic_weighted_kappa(t, x)

# float kappa
def fkappa(t, x):
    t=np.array(t, np.float32)
    x=np.array(x, np.float32)
    u = 0.5 * np.sum(np.square(x - t))
    v = np.dot(np.transpose(x), t - np.mean(t))
    return v / (v + u)

def compute_kappa():
    kappa = fkappa
    opts, args = getopt.getopt(sys.argv[1:], 'fi')
    for o,a in opts:
        if o == "-i":
            kappa = ikappa

    t,x = [],[]
    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            break
        if not line:
            break
        toks = line.split()
        t.append(toks[0])
        x.append(toks[1])
            
    k = kappa(t, x)
    print('{0:0.4}'.format(k))

if __name__ == "__main__":
    compute_kappa()