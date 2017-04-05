import requests
import requests_cache
import json
import operator
import logging

from skimage.io import imread
from cStringIO import StringIO

requests_cache.install_cache('tmdb_cache', backend='sqlite')


class TMDBW:
    """ TMDB wrapper for retrieving movie posters into numpy arrays.

    Usage example::

        from skimage.viewer import ImageViewer
        from tmdb import TMDB
        t = TMDBW("APIKEY")
        poster = t.get_movie("tt0062622")["poster"]
        ImageViewer(poster).show()

    """
    def __init__(self, api_key):
        self.api_key = api_key
        self.service_url = "http://api.themoviedb.org/3"
        self.api_params = {"api_key": self.api_key}
        self.config = json.loads(self._request("/configuration"))

        self.poster_sizes = self.config["images"]["poster_sizes"]
        self.genres = json.loads(self._request("/genre/movie/list", {"language": "en-US"}))
        self.images_base_url = self.config["images"]["base_url"]

    def _request(self, request, params={}):
        params = dict(self.api_params.items() + params.items())
        return requests.get(self.service_url + request, params).content

    def _request_image(self, request, params={}):
        return requests.get(self.images_base_url + request, self.api_params).content

    def get_top_movies(self, release_year=None, limit=10):
        logging.debug("get_top_movies({}, {})".format(release_year, limit))
        params = {
          "primary_release_year": release_year,
          "sort_by": "popularity.desc"
        }

        page = 1
        pages = json.loads(self._request("/discover/movie", params))["total_pages"]
        while page <= pages:
            logging.debug("get_top_movies - fetching page {}".format(page))
            params["page"] = page
            page += 1

            movies = json.loads(self._request("/discover/movie", params))["results"]
            for movie in movies:
                yield self.get_movie(movie["id"])
                limit -= 1
                if limit == 0:
                    return

    def get_movie(self, movie_id, size="original"):
        """Get the movie details.

        :param movie_id: this can either be a TMDB or an IMDB id.
        """
        logging.debug("get_movie({}, {})".format(movie_id, size))
        if size not in self.poster_sizes:
            raise Exception("Poster size {} is not in {}.".format(size, self.poster_sizes))

        movie = json.loads(self._request("/movie/{}".format(movie_id)))
        poster = StringIO(self._request_image("/{}{}".format(size, movie["poster_path"])))
        return {
          "imdb_id": movie["imdb_id"],
          "tmdb_id": movie["id"],
          "adult": movie["adult"],
          "budget": movie["budget"],
          "genres": [g["name"] for g in movie["genres"]],
          "languages": [movie["original_language"]],
          "overview": movie["overview"],
          "poster": imread(poster),
          "release_date": movie["release_date"],
          "revenue": movie["revenue"],
          "runtime": movie["runtime"],
          "tagline": movie["tagline"],
          "title": movie["title"]
        }

    def get_genres(self):
        """Get the list of genres defined on TMDB"""
        genres = json.loads(self._request("/genre/movie/list"))["genres"]
        return [g["name"] for g in genres]
