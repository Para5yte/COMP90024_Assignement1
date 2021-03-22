# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import cell
import json
import multiprocessing as mp
import numpy as np
import time
import math
import os
from mpi4py import MPI
from shapely.geometry import Point, Polygon


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
    with open(file, encoding='utf-8') as json_file:
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


def get_tweet_cell_location(tweet_location, cells):
    print(6)


def get_tweet_sentiment_score(tweet_text, word_dictionary):
    print(5)


# import sys
#
# print('Length of list:', len(sys.argv))
# print(sys.arv)


def main():
    """ main function

    :return:
    """

    # start
    start_time = time.time()

    comm = MPI.COMM_WORLD
    my_rank = comm.Get_rank()
    processors = comm.Get_size()

    # TODO
    # will need to have a way to check nodes as well
    #
    # for dual node processes
    # create a cell {} for each process
    # get roll of json file then irritate through
    # tweets[num_rows % num_process * (rank - 1):num_rows % num_process * Rank]
    # return cells{} to master process and add values together

    cells = {}

    word_dictionary = get_sentiment_dictionary('AFINN.txt')

    melb_grid = get_json_object('melbGrid.json')  # get the melbourne grid json object

    for feature in melb_grid['features']:
        temp_id = feature["properties"]["id"]
        temp_cell = Cell(feature["properties"]["id"])
        temp_array = np.array(feature["geometry"]["coordinates"])  # changes the coordinates to an numpy array
        temp_cell.polygon = Polygon(temp_array[0])
        # cells.append(temp_cell)
        cells[temp_id] = temp_cell

    tweets = get_json_object('tinyTwitter.json')  # Reads the twitter data file
    # tweets = get_json_object('smallTwitter.json')  # Reads the twitter data file

    # total number of tweets in the twitter json file
    total_tweets = len(tweets["rows"])

    # number of tweets this set of script will run
    # TODO
    # currently this will round up the number
    # therefore if we have 19 tweets but 8 processors only 6 of them will do work
    #
    num_tweets = math.ceil(total_tweets / processors)

    tweets = tweets["rows"][num_tweets*my_rank:num_tweets * (my_rank + 1)]

    number_of_tweets = 0
    for tweet in tweets:
        tweet_location = Point(tweet['value']['geometry']['coordinates'])
        # print(tweet_location)
        # print("process: ", my_rank)
        number_of_tweets += 1
        # print("process: ", my_rank, tweet['value']["properties"]["text"])
        # return cell id
        # TODO
        # the below block of code to be a function which returns which cell the tweet was located in

        for cell in cells:
            polygon = cells[cell].polygon
            if tweet_location.within(polygon):
                cell_id = cells[cell].polygon
                cells[cell].num_tweet += 1
                break
            # elif tweet_location.touches(polygon):

            # elif tweet_location.touches(polygon):
            # TODO
            # print(tweet_location.intersects(polygon))
            # print(tweet_location.touches(polygon))
            # print("check overlap ", cell.id)

        # return sentiment score
        # tweet_text = tweet['value']['properties']['text']

    print("process", my_rank, "number of tweets this process went through", number_of_tweets)

    if my_rank != 0:
        message = "hello from" + str(my_rank)
        comm.send(cells, dest=0)
    else:
        print("--- %s seconds ---" % (time.time() - start_time))
        for proc_id in range(1, processors):
            cell_info = comm.recv(source=proc_id)

            # TODO
            # create a function for the below block of code
            # the block of code below will add all the returned vale from other process
            for cell in cell_info:
                cells[cell].num_tweet += cell_info[cell].num_tweet
                cells[cell].sentiment_score += cell_info[cell].sentiment_score

        for cell in cells:
            print("number of tweets in", cell, cells[cell].num_tweet)

            # print("process 0 receives message from process", proc_id, ":", message)


if __name__ == '__main__':
    main()
