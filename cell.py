from shapely.geometry import Point, Polygon


class Cell:
    # Number of Tweets given in a particular cell
    numTweet = 0

    # Overall Sentiment Score
    sentimentScore = 0

    # Polygon coords
    polygon = Polygon()

    # way to initiate a class with input

    def __init__(self, name):
        self.name = name