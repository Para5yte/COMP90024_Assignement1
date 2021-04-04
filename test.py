words = ["i'm", 'at', '@rivastkilda', 'in', 'st', 'kilda,']

for word in words:
    if word == "at":
        new_word = "cum"
        index = words.index(word)
        words.insert(index+1, new_word)

print(words)

test = {"key1" : 1, "key2" : 3}
print(test.get("key3",0))
test["key3"]

def get_tweet_sentiment_score(tweet_text):
    """ get the tweet sentiment score from the AFINN dictionary

    :param tweet_text:

    :return:
    """
    split_text = tweet_text.lower().split()

    score = 0
    temp_score = 0
    temp_word = ""
    new_word = ""

    print(split_text)

    for word in split_text:

        # TODO happy?:-) matches as happy? is a valid substring but happy:-)
        #  does not fit as : is not a valid punctuation symbol but happy :-) is a
        # if a word contains the punctuation then it's a match
        # more examples of matches: good!@
        # Good!jhkajshkjhads
        # Good?,mkkwjwh
        # won can match won't
        """ when a word ends with one of the punctuation ! , ? ' " the word ends at that point
            therefore there is no need to check in AFINN with the word following it
        """

        #   if word.endswith(punctuation_tuple):    #thold code for ends with but richard said it's not right
        if any(punctuation in punctuation_tuple for punctuation in word):

            # removes all the punctuation of a word from the back
            word = remove_punctuation(word)

            # check if current word match something e.g. we don't want won't matching to won

            if temp_word != "":
                try:
                    temp_score = afinn_dictionary[temp_word + " " + word]
                    score += temp_score
                except KeyError:
                    if temp_score == 0:
                        temp_word = ""
                        continue

                    score += afinn_dictionary[temp_word]
                    temp_word = ""
                finally:
                    temp_score = 0
            else:
                try:
                    score += afinn_dictionary[word]
                    print(word)
                except KeyError:

                    # check for words in afinn starting with word e.g. "can't" afinn includes "can't stand"
                    if word_beginning_with(word):
                        try:
                            temp_score = afinn_dictionary[word]
                        except KeyError:
                            None
                        finally:
                            temp_word += word
                    else:
                        try:
                            score += afinn_dictionary[temp_word]
                        except KeyError:
                            None
                        finally:
                            temp_word = ""
                    # split the word into
                    # also has to check if a word contains 2 words e.g cool!good
                    # index = words.index(word)
                    # words.insert(index+1, new_word)
                    continue

        else:
            # check if the temporary word is empty, if not search in AFINN
            if temp_word != "":
                temp_word += " " + word

                if word_beginning_with(temp_word):
                    try:
                        temp_score = afinn_dictionary[temp_word]
                    except KeyError:
                        continue
                else:
                    try:
                        score += afinn_dictionary[temp_word]
                    except KeyError:
                        if temp_score != 0:
                            score += temp_score
                            temp_score = 0
                    finally:
                        temp_word = ""

            else:
                temp_word += word

                if word_beginning_with(temp_word):
                    try:
                        temp_score = afinn_dictionary[temp_word]
                    except KeyError:
                        continue
                else:
                    try:
                        score += afinn_dictionary[temp_word]
                    except KeyError:
                        None
                    finally:
                        temp_word = ""

### TODO can delete block of code
def filter_list_of_dict(key, list_of_dict):
    """ Filters a list of dict only keeping the given key of each line

    :param key: str
        key to filter by
    :param list_of_dict:[]
        list of dictionary

    :return: []
        returns the new list which only keeps the key
    """
    new_list = []
    for line in list_of_dict:
        new_list.append(line[key])
    return new_list