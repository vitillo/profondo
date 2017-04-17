import argparse
import logging
import pandas as pd
import numpy as np
import tables

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

    start_time = time()
    tmdb = TMDBW(args.api_key)
    data = []
    image_export = tables.open_file("image_export.h5", mode="w")
    image_shape = list(tmdb.get_top_movies(limit=1).next()["poster"].shape)
    images = image_export.create_earray(image_export.root, 'images',
        tables.Float32Atom(), [0] + image_shape, "images")

    for movie in tmdb.get_top_movies(limit=args.limit):
        images.append(movie["poster"].reshape([1] + image_shape))
        del movie["poster"]
        data.append(movie)

    pd.DataFrame(data).to_pickle("metadata_export")
    image_export.close()
    end_time = time()

    print "Data export completed in {} seconds.".format(int(end_time - start_time))
