# i made a change here

# Press Shift+F10 to execute it or replace it with your code.

import json
import numpy as np
import time
import math
import os
import sys
from mpi4py import MPI
from shapely.geometry import Point, Polygon
from collections import Counter


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

    def __init__(self, id):
        """

        :param id: str
            the id of the grid cell
        """
        self.id = id


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

    temp = {}
    with open(file) as f:
        for line in f:
            # print(line)

            # this block of code works without assumption
            split_line = line.split()
            if len(split_line) > 2:
                (key, val) = (" ".join(split_line[:len(split_line) - 1]), split_line[-1])
            else:
                (key, val) = split_line

            # this code below only works if we assume the text file will be indented by "word" \t "score"
            # (key, val) = line.split('\t', 1)
            temp[key] = int(val)

    return temp

def get_repeated_cells():
    count = {}
    for s in check_string:
        if s in count:
            count[s] += 1
        else:
            count[s] = 1

    for key in count:
        if count[key] > 1:
            print
            key, count[key]


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


def get_tweet_sentiment_score(tweet_text, word_dictionary):
    print(5)


def main(argv):
    """ main function

    :return:
    """

    # start the timer
    start_time = time.time()

    comm = MPI.COMM_WORLD           # initialise MPI
    my_rank = comm.Get_rank()       # gets the rank of current process
    processors = comm.Get_size()    # how many processors where allocated

    # get the input twitter json file
    tweets = get_json_object(argv[1])

    """
    Block of code below uses broadcasting as a method to read all given  json files
    then distribute them among all processes, however during testing it was found that 
    the program runs faster if you read the json file by it self
    """
    if my_rank == 0:
        word_dictionary = get_sentiment_dictionary('AFINN.txt')
        melb_grid = get_json_object('melbGrid.json')
    else:
        word_dictionary = None
        melb_grid = None
    word_dictionary = comm.bcast(word_dictionary, root=0)     # get the list of words with a score
    melb_grid = comm.bcast(melb_grid, root=0)                 # get the melbourne grid json object

    # TODO will need to use broadcast for bigTwitter file
    """
    As Richard quoted
    The main challenge is to read/process a big file and have 8 processes running in parallel to do this.
     No single process should read in all of the data into memory

    Take smallTwitter.json and have each process (master/slave) running and processing 
    “parts” of the big file

    """

    # cells infomation of this process (key, object) -> ("A1", cell)
    cells = {}

    # will read melbourne grid json and append into cell dictionary
    for feature in melb_grid['features']:
        temp_id = feature["properties"]["id"]
        temp_cell = Cell(feature["properties"]["id"])
        temp_array = np.array(feature["geometry"]["coordinates"])  # changes the coordinates to an numpy array
        temp_cell.polygon = Polygon(temp_array[0])
        # cells.append(temp_cell)
        cells[temp_id] = temp_cell

    # total number of tweets in the twitter json file
    total_tweets = len(tweets["rows"])

    # number of tweets this set of script will run
    # TODO
    # currently this will round up the number
    # therefore if we have 19 tweets but 8 processors only 6 of them will do work
    num_tweets = math.ceil(total_tweets / processors)

    tweets = tweets["rows"][num_tweets*my_rank:num_tweets * (my_rank + 1)]

    number_of_tweets = 0
    for tweet in tweets:
        tweet_location = Point(tweet['value']['geometry']['coordinates'])
        # print(tweet_location)
        # print("process: ", my_rank)

        # TODO delete below line as it's for testing
        number_of_tweets += 1

        # get cell id in which the tweet occurred
        cell_id = get_tweet_cell_location(tweet_location, cells)
        cells[cell_id].num_tweet += 1

        # TODO
        # return sentiment score
        # tweet_text = tweet['value']['properties']['text']

    print("process", my_rank, "number of tweets this process went through", number_of_tweets)

    if my_rank != 0:

        comm.send(cells, dest=0)
    else:

        for proc_id in range(1, processors):
            cell_info = comm.recv(source=proc_id)
            # TODO
            # create a function for the below block of code
            # the block of code below will add all the returned vale from other process
            for cell in cell_info:
                cells[cell].num_tweet += cell_info[cell].num_tweet
                cells[cell].sentiment_score += cell_info[cell].sentiment_score

        # for cell in cells:
            # print("number of tweets in", cell, cells[cell].num_tweet)

        with open("result.txt", "a") as text_file:
            print("number of processes", processors, file=text_file)
            time_taken = time.time() - start_time
            print("--- %s seconds ---" % time_taken, file=text_file)
        print("--- %s seconds ---" % time_taken)


if __name__ == '__main__':
    main(sys.argv)
