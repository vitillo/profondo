import argparse
import logging
import pandas as pd
import numpy as np

from pandas.io.json import json_normalize
from tmdbw import TMDBW
from time import time


if __name__ == "__main__":
    parser = argparse.ArgumentParser("TMDB Data Exporter")
    parser.add_argument("--limit", help="Number of movies to download", type=int, default=10000)
    parser.add_argument("--logging", help="Enable logging", type=bool, default=False)
    parser.add_argument("--api-key", help="TMDB API key", type=str, required=True)
    args = parser.parse_args()

    if args.logging:
        logging.getLogger().setLevel(logging.DEBUG)

    tmdb = TMDBW(args.api_key)
    data = []
    images = []
    start_time = time()

    for movie in tmdb.get_top_movies(limit=args.limit):
        images.append(movie["poster"])
        del movie["poster"]
        data.append(movie)

    pd.DataFrame(data).to_pickle("metadata_export")
    np.save("image_export", np.array(images))
    end_time = time()

    print "Data export completed in {} seconds.".format(int(end_time - start_time))
