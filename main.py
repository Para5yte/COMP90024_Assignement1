# COMP90024 - Cluster and Cloud Computing Assignment 1
# Takemitsu Yamanaka 757038
# Barbara Montt    1017615

import ujson as json
import re
import mmap
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


def get_json_object(file_path):
    """ reads the json file and returns the json object

    :param file_path: str
        path to the json file
    :return: loaded json object
    """
    with open(os.path.realpath(file_path), encoding='utf-8') as json_file:
        return json.load(json_file)


def get_sentiment_dictionary(file_path):
    """ reads the txt file and returns a dictionary of words with related score

    :param file_path: str
        path to the sentiment of the word file
    :return:    {}
        dictionary of word (key) and it's related score (value)
    """

    dictionary = {}

    with open(os.path.realpath(file_path)) as file:
        for line in file:

            # assume the text file will be indented by "word" \t "score"
            (key, val) = line.split('\t', 1)

            # preprocess any key with two words and append the Value True to them
            # e.g. "cool stuff", create new key for "cool " append to True to it
            # e.g. "does not work", create 2 new key for "does " & "does not " and append True to it
            if ' ' in key:
                keys = key.split(' ')
                for i in range(len(keys) - 1):
                    new_key = keys[i] + ' '
                    dictionary[new_key] = True

            dictionary[key] = int(val)

    return dictionary


def get_cells(melb_grid):
    """     create a cells dictionary {cell_id : cell_class}

    :param melb_grid: jsonObject
        melbourne grid json object
    :return: {}
        returns the cells dictionary
    """
    cells = {}

    for feature in melb_grid['features']:
        temp_id = feature["properties"]["id"]
        temp_cell = Cell(feature["properties"]["id"])               # initialise a new cell class
        temp_array = np.array(feature["geometry"]["coordinates"])  # changes the coordinates to an numpy array
        temp_cell.polygon = Polygon(temp_array[0])
        cells[temp_id] = temp_cell

    return cells


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


def word_beginning_with(word, afinn_dictionary):
    """ Check if the dictionary has any keys starting with the input word
        e.g. "cool stuff" with "cool" as input for word, then return true

    :param word: str
        word to search
    :param afinn_dictionary: {}
        afinn dictionary
    :return: bool
        true if there is a key which starts with word
        false if there isn't a key which starts with word
    """
    word += ' '

    return afinn_dictionary.get(word, False)


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

    for i in reversed(range(len(word))):
        if word[i] in punctuation_tuple:
            word = word[:i]
        else:
            break
    return word


def get_tweet_sentiment_score(tweet_text, afinn_dictionary):
    """ get the tweet sentiment score from the AFINN dictionary

    :param tweet_text: str
        tweet text
    :param afinn_dictionary:
        afinn_dictionary
    :return: int
        sentiment score of the tweet
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
                temp_word = '%s %s' % (temp_word, word)

                # check if there are any matches beginning with temp word
                if word_beginning_with(temp_word, afinn_dictionary):
                    temp_score = afinn_dictionary.get(temp_word, 0)

                else:

                    # check if the temporary word is in AFINN, if not get the score of current word
                    # e.g. temp_word = "cool stuff" returns 3
                    # e.g. temp_word = "cool corpse" returns -1
                    if afinn_dictionary.get(temp_word, 0) == 0:
                        if word_beginning_with(word, afinn_dictionary):
                            temp_word = word
                            continue
                        score += afinn_dictionary.get(word, 0)
                    else:
                        score += afinn_dictionary.get(temp_word, 0)
                        temp_word = ""

                    # using example above, if temp_word = "cool corpse"
                    # we'll use the stored temp_score for "cool"
                    if temp_score != 0:
                        score += temp_score
                        temp_score = 0

            # if temporary word is empty, then check if there is a match or not
            else:
                temp_word += word

                # check if there are any matches beginning with temp word
                if word_beginning_with(temp_word, afinn_dictionary):
                    temp_score = afinn_dictionary.get(temp_word, 0)

                else:
                    score += afinn_dictionary.get(temp_word, 0)
                    temp_word = ""

    return score


def print_output_file(filename, results_cell, result_time):
    """ print the results into a file with the name Filename

    :param filename: str
        output filename
    :param results_cell: {} (cell_id, cell class)
         results all of the cells from all processes
    :param result_time: float
        time taken for the script to run
    """

    with open(filename, "w") as text_file:
        print("Cell\t #Total Tweets\t #Overal Sentiment Score", file=text_file)

        # TODO we can also change the printing using a grid or something
        for cell in results_cell:
            print("%s \t\t %d \t\t %d" % (results_cell[cell].id,
                results_cell[cell].num_tweet, results_cell[cell].sentiment_score), file=text_file)

        print("time taken for this script to run --- %f seconds ---"
              % result_time, file=text_file)


def main(argv):
    """ main function

    :return:
    """

    # start the timer
    start_time = time.time()

    comm = MPI.COMM_WORLD           # initialise MPI
    my_rank = comm.Get_rank()       # gets the rank of current process
    processors = comm.Get_size()    # how many processors where allocated

    # we can use the following for name convention for our output files
    # https://www.tutorialspoint.com/python/python_command_line_arguments.htm

    """
        if current rank is 0 (master process)
        process all the file reading and data transformation
    """
    if my_rank == 0:

        # initialise the afinn_dictionary global variable
        afinn_dictionary = get_sentiment_dictionary('AFINN.txt')

        # process melbourne grid object into a cell dictionary
        cells = get_cells(get_json_object('melbGrid.json'))

    else:
        cells = None
        afinn_dictionary = None

    # broadcast afinn dictionary to all other processors
    afinn_dictionary = comm.bcast(afinn_dictionary, root=0)

    # cells information of this process (key, object) -> ("A1", cell)
    # broadcast the cells information to all other processes
    cells = comm.bcast(cells, root=0)

    twitter_filepath = argv[1]

    with open(os.path.realpath(twitter_filepath), mode="r", encoding="utf8") as file_obj:
        # open the file has a mmap
        twitter_mmap = mmap.mmap(file_obj.fileno(), 0, access=mmap.ACCESS_READ)

        for index, tweet in enumerate(iter(twitter_mmap.readline, b'')):
            if index % processors != my_rank:
                continue

            # reconstruct the json line
            # assumption made, twitters.json lines particularly lines which include tweets end with with ',\r\n'
            # the last line of a twitter.json file can end with ']}\r\n' as seen in tinyTwitter.json
            tweet = re.sub('(]}|,)\r\n', '', tweet.decode())

            print(tweet)

            # validate the json line
            # for the first row of the twitter files and sometimes the last row
            try:
                tweet = json.loads(tweet)
            except ValueError:
                continue

            tweet = tweet['value']

            # get cell id in which the tweet occurred
            tweet_location = Point(tweet['geometry']['coordinates'])
            cell_id = get_tweet_cell_location(tweet_location, cells)

            if cell_id is None:
                continue

            cells[cell_id].num_tweet += 1

            tweet_text = tweet['properties']['text']
            score = get_tweet_sentiment_score(tweet_text, afinn_dictionary)
            cells[cell_id].sentiment_score += score

        twitter_mmap.close()

    if my_rank != 0:
        comm.send(cells, dest=0)
    else:
        for proc_id in range(1, processors):
            cell_info = comm.recv(source=proc_id)

            # combine the data from other processes
            for cell in cell_info:
                cells[cell].num_tweet += cell_info[cell].num_tweet
                cells[cell].sentiment_score += cell_info[cell].sentiment_score

        # get time taken to process twitter file
        time_taken = time.time() - start_time

        try:
            results_filename = argv[2]
        except IndexError:
            results_filename = "results.txt"

        # output the result of the score for each cell and the number tweets in the cell
        # with the time taken to run the script
        print_output_file(results_filename, cells, time_taken)


if __name__ == '__main__':
    main(sys.argv)
