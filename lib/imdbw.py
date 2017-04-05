import imdb
import logging
import requests
import requests_cache

from skimage.io import imread
from cStringIO import StringIO

requests_cache.install_cache('imdb_cache', backend='sqlite')

class IMDBW:
    """ IMDB wrapper for retrieving movie metadata and posters.

    The documentation for the underlying IMDbPy API can be found
    at http://imdbpy.sourceforge.net/docs/README.package.txt

    Usage example::

        imdb = IMDBW()
        poster = t.get_movie("0062622")["poster"]
        ImageViewer(poster).show()

    """
    def __init__(self):
        # TODO: Allow for local storage through SQL. This would
        #       require manually feeding in the database the plain
        #       text data downloaded from IMDB.
        self._imdb = imdb.IMDb()

    def get_top_movies(self, limit=10):
        """Get the top movies."""
        logging.debug("get_top_movies({})".format(limit))
        top_movies = self._imdb.get_top250_movies()

        for movie in top_movies[:limit]:
            yield self.get_movie(movie.movieID)

    def search_movie(self, title):
        """Search a movie by the title.

        This returns a {<movie_id>: <movie title>, ...} dictionary
        with the results.
        """
        logging.debug("search_movie({})".format(title))

        results = self._imdb.search_movie(title)

        return {movie.movieID: movie.get("long imdb title") for movie in results}

    def get_movie(self, movie_id):
        movie = self._imdb.get_movie(movie_id)

        is_adult_movie = "Adult" in movie.get("genres", [])

        # Fetch the full-size poster from IMDB.
        cover_url = movie.get("full-size cover url", None)
        poster_data = None
        if cover_url:
            data = StringIO(requests.get(cover_url).content)
            poster_data = imread(data)

        return {
          "imdb_id": "tt{}".format(movie_id),
          "adult": is_adult_movie,
          "assistant_directors": [d.get("name") for d in movie.get("assistant director", [])],
          "cast": [c.get("name") for c in movie.get("cast", [])],
          "directors": [d.get("name") for d in movie.get("director", [])],
          "genres": movie.get("genres", []),
          "languages": movie.get("languages", []),
          "overview": movie.get("plot outline"),
          "plot": movie.get("plot"),
          "poster": poster_data,
          "producers": [p.get("name") for p in movie.get("producer", [])],
          "release_date": movie.get("year"),
          "title": movie.get("title"),
          "visual_effects": [d.get("name") for d in movie.get("visual effects", [])],
          "writers": [w.get("name") for w in movie.get("writer", [])]
        }
