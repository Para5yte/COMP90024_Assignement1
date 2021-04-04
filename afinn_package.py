import math
import re

# sys.setdefaultencoding('utf-8')

# AFINN-111 is as of June 2011 the most recent version of AFINN
filenameAFINN = 'AFINN.txt'
# afinn = dict(map(lambda (w, s): (w, int(s)), [
#     ws.strip().split('\t') for ws in open(filenameAFINN)]))
afinn_dictionary = {}

with open(filenameAFINN) as f:
    for line in f:
        split_line = line.split()
        if len(split_line) > 2:
            (key, val) = (" ".join(split_line[:len(split_line) - 1]), split_line[-1])
        else:
            (key, val) = split_line
        afinn_dictionary[key] = int(val)



def sentiment(text):
    """
    Returns a float for sentiment strength based on the input text.
    Positive values are positive valence, negative value are negative valence.
    """
    words = text.lower().split()
    print(words)
    sentiments = list(map(lambda word: afinn_dictionary.get(word, 0), words))
    score = 0
    if sentiments:

        score = sum(sentiments)
        # # How should you weight the individual word sentiments?
        # # You could do N, sqrt(N) or 1 for example. Here I use sqrt(N)
        # sentiment = float(sum(sentiments)) / math.sqrt(len(sentiments))

    else:
        score = 0

    return score


if __name__ == '__main__':
    # # Single sentence example:
    # text = "Finn is stupid and idiotic"
    # print("%6.2f %s" % (sentiment(text), text))
    #
    # # No negation and booster words handled in this approach
    # text = "Finn is only a tiny bit stupid and not idiotic"
    # print("%6.2f %s" % (sentiment(text), text))

    # No negation and booster words handled in this approach
    text = "Good guidance for all you ideas people coming into m2m, once-in-a-lifetime does not work"
    print("%6.2f %s" % (sentiment(text), text))
