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

        imdb = IMDB()
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
          "genres": movie.get("genres", []),
          "poster": poster_data,
          "adult": is_adult_movie,
          "budget": None,
          "languages": movie.get("languages", []),
          "title": movie.get("title"),
          "overview": movie.get("plot"),
          "tagline": None,
          "release_date": movie.get("year"),
          "revenue": None
        }
