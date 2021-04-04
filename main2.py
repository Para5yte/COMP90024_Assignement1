# COMP90024 - Cluster and Cloud Computing Assignment 1
# Takemitsu Yamanaka 757038
# Barbara Montt    1017615

import json
import numpy as np
import time
import os
import sys
from mpi4py import MPI
from shapely.geometry import Point, Polygon
from collections import Counter

""" Global Variables
"""
punctuation_tuple = ('!', ',', '?', '.', "'", '"')
afinn_dictionary = {}


class Cell:
    """
    A class used to represent a grid cell

    ...

    Attributes
    ----------
    num_tweet : int
        number of tweets tweeted in a particular cell

    sentiment_score : int
        overall sentiment score

    polygon : list array of coordinates
        polygon coordinates
    """

    num_tweet = 0
    sentiment_score = 0
    polygon = Polygon()

    def __init__(self, cell_id):
        """

        :param cell_id: str
            the id of the grid cell
        """
        self.id = cell_id


def get_json_object(file):
    """ reads the json file and returns the json object

    :param file: str
        path to the json file
    :return: loaded json object
    """
    with open(os.path.realpath(file), encoding='utf-8') as json_file:
        return json.load(json_file)


def get_sentiment_dictionary(file):
    """ reads the txt file and returns a dictionary of words with related score

    :param file: str
        path to the sentiment of the word file
    :return:    {}
        dictionary of word (key) and it's related score (value)
    """

    with open(file) as f:
        for line in f:

            # this block of code works without assumption
            split_line = line.split()
            if len(split_line) > 2:
                (key, val) = (" ".join(split_line[:len(split_line) - 1]), split_line[-1])
            else:
                (key, val) = split_line

            # this code below only works if we assume the text file will be indented by "word" \t "score"
            # (key, val) = line.split('\t', 1)
            afinn_dictionary[key] = int(val)



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


def get_tweet_cell_location(tweet_location, cells):
    """ get the cell id of the location where the tweet occurred


    :param tweet_location: Point(x, y)
        Shapely Point of x and y coordinate of where the tweet occurred
    :param cells: {} (key, object) -> ("A1", cell)
        cell grid dictionary of cell classes
    :return: str
        cell_id of where the tweet occurred
    """

    # if a tweet were to occur in between any grid cells, it'll append it's cell id to the below list
    borders_of_tweet = []
    for cell in cells:
        polygon = cells[cell].polygon
        if tweet_location.within(polygon):
            return cells[cell].id

        elif tweet_location.touches(polygon):
            borders_of_tweet.append(cells[cell].id)

    # if tweet was tweeted right on the border of edge of melbourne grid
    if len(borders_of_tweet) == 1:
        return borders_of_tweet[0]

    # if tweet intersects between 2 cells
    elif len(borders_of_tweet) == 2:

        # prioritise cell on the left, e.g. if tweet occurs between A1/A2 then return A1
        if borders_of_tweet[0][0] == borders_of_tweet[1][0]:
            return min(borders_of_tweet)

        # prioritise cell below, e.g. if tweet occurs between A1/B1 then return B1
        if borders_of_tweet[0][1] == borders_of_tweet[1][1]:
            return max(borders_of_tweet)

    # if tweet intersects between 3 cells
    elif len(borders_of_tweet) == 3:
        # get the prefix of each cell id
        cell_prefix = [x[0] for x in borders_of_tweet]
        # count the occurrence of each prefix
        counter = Counter(cell_prefix)
        # get the most occurred cell id prefix
        most_occurrence_cell_predix = max(counter, key=counter.get)
        # get a list of cells with the most prefix
        more_than_once = list(filter(lambda x: most_occurrence_cell_predix in x, borders_of_tweet))

        # prioritise cell below or left, e.g. if tweet occurs between C2, C3, D4 then return C2
        return min(more_than_once)

    # if tweet intersects between 4 cells then return the left bottom cell which is the 3rd cell id
    elif len(borders_of_tweet) == 4:
        return sorted(borders_of_tweet)[2]

    else:
        return None


def word_beginning_with(word):
    """ Check if the dictionary has any keys starting with the input word
        e.g. "cool stuff" with "cool" as input for word, then return true

    :param word: str
        word to search
    :param dictionary: {}
        AFINN dictionary
    :return: bool
        true if there is a key which starts with word
        false if there isn't a key which starts with word
    """
    word += " "
    for key in afinn_dictionary.keys():
        if key.startswith(word):
            return True
    return False


def remove_punctuation(word):
    """ removes the punctuation on the end of the word
        e.g. "awesome!!" will return "awesome"
        e.g. "awesome!@" will return "awesome!@"
        e.g. "awesome@!" will return "awesome@"

    :param word: str
        word to remove punctuation
    :return: str
        returns the edited word
    """
    new_word = ""
    for i in reversed(range(len(word))):
        if word[i] in punctuation_tuple:
            word = word[:i]
        else:
            break
    return word


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



        # check if current word has any punctuation
        if any(punctuation in punctuation_tuple for punctuation in word):

            # removes all the punctuation of a word from the back
            word = remove_punctuation(word)

            # if the temp word is not empty try find a match in AFINN
            if temp_word != "":
                # check the score of temp word concatenated with current word
                temp_score = afinn_dictionary.get(temp_word+" "+word, 0)

                if temp_score != 0:
                    score += temp_score
                else:
                    score += afinn_dictionary.get(temp_word, 0)
                    temp_word = ""

                temp_score = 0

            else:
                score += afinn_dictionary.get(word, 0)


                # # check for words in afinn starting with word e.g. "can't" afinn includes "can't stand"
                # if word_beginning_with(word):
                #     try:
                #         temp_score = afinn_dictionary[word]
                #     except KeyError:
                #         None
                #     finally:
                #         temp_word += word
                # else:
                #     try:
                #         score += afinn_dictionary[temp_word]
                #     except KeyError:
                #         None
                #     finally:
                #         temp_word = ""
                # # split the word into
                # # also has to check if a word contains 2 words e.g cool!good
                # # index = words.index(word)
                # # words.insert(index+1, new_word)
                # continue
        else:
            # check if the temporary word is empty, if not search in AFINN
            if temp_word != "":

                temp_word += " " + word

                # check if there are any matches beginning with temp word
                if word_beginning_with(temp_word):

                    temp_score = afinn_dictionary.get(temp_word, 0)

                else:

                    # check if the temporary word is in AFINN, if not get the score of current word
                    # e.g. temp_word = "cool stuff" returns 3
                    # e.g. temp_word = "cool corpse" returns -1
                    if afinn_dictionary.get(temp_word, 0) == 0:

                        print(word, afinn_dictionary.get(word, 0))
                        score += afinn_dictionary.get(word, 0)
                    else:
                        print(temp_word, afinn_dictionary.get(temp_word, 0))
                        score += afinn_dictionary.get(temp_word, 0)

                    # using example above, if temp_word = "cool corpse"
                    # we'll use the stored temp_score for "cool"
                    if temp_score != 0:
                        score += temp_score
                        temp_score = 0

                    temp_word = word

            # if temporary word is empty, then check if there is a match or not
            else:
                temp_word += word

                # check if there are any matches beginning with temp word
                if word_beginning_with(temp_word):
                    temp_score = afinn_dictionary.get(temp_word, 0)

                else:
                    score += afinn_dictionary.get(temp_word, 0)
                    temp_word = ""

    print(split_text)
    print(score)
    return score


def main(argv):
    """ main function

    :return:
    """

    # start the timer
    start_time = time.time()

    comm = MPI.COMM_WORLD  # initialise MPI
    my_rank = comm.Get_rank()  # gets the rank of current process
    processors = comm.Get_size()  # how many processors where allocated

    """
    Block of code below uses broadcasting as a method to read all given json files
    then distribute them among all processes
    """
    if my_rank == 0:
        melb_grid = get_json_object('melbGrid.json')
    else:
        melb_grid = None
    melb_grid = comm.bcast(melb_grid, root=0)  # get the melbourne grid json object

    # initialise the afinn_dictionary global variable
    get_sentiment_dictionary('AFINN.txt')

    # TODO explain to Babara
    """
    As Richard quoted
    The main challenge is to read/process a big file and have 8 processes running in parallel to do this.
     No single process should read in all of the data into memory

    Take smallTwitter.json and have each process (master/slave) running and processing 
    “parts” of the big file
    
    word_dictionary = get_sentiment_dictionary('AFINN.txt')
    #print (word_dictionary)
    melb_grid = get_json_object('melbGrid.json')  # get the melbourne grid json object

    """

    # we can use the following for name convention for our output files
    # https://www.tutorialspoint.com/python/python_command_line_arguments.htm

    # scatter the tweet data in chucks
    if my_rank == 0:
        # get the input twitter json file
        tweets = get_json_object(argv[1])

        # only need the tweet data which is stored in the key "rows"
        tweets = tweets["rows"]

        """
            filters the tweet list to only keep data which we'll be analysing such as 
            geometry and text properties which is stored in the key 'value'
            This in term will also save memory (RAM)
            code taken from https://stackoverflow.com/questions/25148611/how-do-i-extract-all-the-values-of-a-specific-key-from-a-list-of-dictionaries
        """
        tweets = [tweets['value'] for tweets in tweets]

        # divide the tweet into chucks to scatter among other processors
        chunks_of_tweets = [[] for _ in range(processors)]
        for i, tweet in enumerate(tweets):
            chunks_of_tweets[i % processors].append(tweet)

    else:
        tweets = None
        chunks_of_tweets = None

    # scatter the tweet data to save memory for each process therefore processing "parts" of data
    tweets = comm.scatter(chunks_of_tweets, root=0)

    # cells information of this process (key, object) -> ("A1", cell)
    cells = {}
    
    # TODO low prior can make below code a function
    # will read melbourne grid json and append into cell dictionary
    for feature in melb_grid['features']:
        temp_id = feature["properties"]["id"]
        temp_cell = Cell(feature["properties"]["id"])
        temp_array = np.array(feature["geometry"]["coordinates"])  # changes the coordinates to an numpy array
        temp_cell.polygon = Polygon(temp_array[0])
        cells[temp_id] = temp_cell

    number_of_tweets = 0

    for tweet in tweets:

        # get cell id in which the tweet occurred
        tweet_location = Point(tweet['geometry']['coordinates'])
        cell_id = get_tweet_cell_location(tweet_location, cells)

        if cell_id is None:
            continue

        cells[cell_id].num_tweet += 1

        tweet_text = tweet['properties']['text']

        score = get_tweet_sentiment_score(tweet_text)
        cells[cell_id].sentiment_score += score

    if my_rank != 0:
        comm.send(cells, dest=0)
    else:

        for proc_id in range(1, processors):
            cell_info = comm.recv(source=proc_id)

            # combine the data from other processes
            for cell in cell_info:
                cells[cell].num_tweet += cell_info[cell].num_tweet
                cells[cell].sentiment_score += cell_info[cell].sentiment_score

        # TODO less prior, to make the below code a function
        # output the result of the score for each cell and the number tweets in the cell
        # with the time taken to run the script
        # TODO we can also change the printing using a grid or something
        with open("result.txt", "w") as text_file:
            print("Cell\t #Total Tweets\t #Overal Sentiment Score", file=text_file)
            for cell in cells:
                print(cells[cell].id, "\t\t", cells[cell].num_tweet, "\t\t",
                      cells[cell].sentiment_score, file=text_file)

            time_taken = time.time() - start_time
            print("time taken for this script to run with %s Processors --- %s seconds ---"
                  % (processors, time_taken), file=text_file)
            print("time taken for this script to run with %s Processors --- %s seconds ---"
                  % (processors, time_taken))


if __name__ == '__main__':
    main(sys.argv)
