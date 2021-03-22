# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# import cell
import json
import multiprocessing as mp
import numpy as np
import time
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

    # way to initiate a class with input

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


def get_cell_id():

    print(6)
# import sys
#
# print('Length of list:', len(sys.argv))
# print(sys.arv)


def main():
    """ main function

    :return:
    """

    cells = {}

    melb_grid = get_json_object('melbGrid.json')  # get the melbourne grid json object

    for feature in melb_grid['features']:
        temp_id = feature["properties"]["id"]
        temp_cell = Cell(feature["properties"]["id"])
        temp_array = np.array(feature["geometry"]["coordinates"])  # changes the coordinates to an numpy array
        temp_cell.polygon = Polygon(temp_array[0])
        #cells.append(temp_cell)
        cells[temp_id] = temp_cell

    # if Tweet is within grid
    # Work out the grid side
    # A1,B2,C3
    #
    # print(point.within(polygon))
    #print(temp.polygon.contains(point))
    #print(temp.polygon.touches(point))

    # for obj in a:
    #     print(obj.name)
    #     print(obj.polygonCoords)
    #       if polygon.touchs(point):


    tweets = get_json_object('tinyTwitter.json')  # Reads the twitter data file
    #tweets = get_json_object('smallTwitter.json')  # Reads the twitter data file

    for tweet in tweets["rows"]:
        tweet_location = Point(tweet['value']['geometry']['coordinates'])
        #print(tweet_location)

        # return cell id
        for cell in cells:
            polygon = cell.polygon
            if tweet_location.within(polygon):
                print(cell.id)
                cell_id = cell.id
                break
            elif tweet_location.touches(polygon):
                #TODO
                print(tweet_location.intersects(polygon))
                print(tweet_location.touches(polygon))
                print("check overlap ", cell.id)

        #return sentiment score



        tweet_text = tweet['value']['properties']['text']
        #print(tweet_text)


if __name__ == '__main__':
    main()


print("Number of processors: ", mp.cpu_count())

