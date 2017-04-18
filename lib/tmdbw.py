import requests
import requests_cache
import json
import operator
import logging
import traceback
import time

from skimage import transform
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

    def _request(self, request, params={}, base=None):
        if base is None:
            base = self.service_url

        while True:
            params = dict(self.api_params.items() + params.items())
            req = requests.get(base + request, params)
            if req.status_code == 429:
                retry_after = req.headers["Retry-After"]
                logging.debug("_request({}) - retrying after {} s".format(request, retry_after))
                time.sleep(int(retry_after) + 1)
                continue
            return req.content

    def _request_image(self, request, params={}):
        return self._request(request, params=params, base=self.images_base_url)

    def _get_top_movies(self, genre_id):
        logging.debug("_get_top_movies({})".format(genre_id))
        params = {
          "with_genres": genre_id,
          "sort_by": "popularity.desc"
        }

        page = 1
        pages = min(json.loads(self._request("/discover/movie", params))["total_pages"], 1000)
        movie_num = 0
        while page <= pages:
            logging.debug("_get_top_movies({}) - fetching page {}".format(genre_id, page))
            params["page"] = page
            page += 1

            movies = json.loads(self._request("/discover/movie", params))["results"]
            for movie in movies:
                try:
                    logging.debug("_get_top_movies({}) - fetching movie {}".format(genre_id, movie_num))
                    yield self.get_movie(movie["id"])
                    movie_num += 1
                except Exception:
                    logging.warning("_get_top_movies({}) - failed to fetch movie {}".format(genre_id, movie_num))
                    traceback.print_exc()

    def get_top_movies(self, limit):
        # This unconvient way of fetching data is needed to avoid hitting the
        # maximum page limit of 1000 and to heuristically balance the data.
        cursors = {g: [g, 0, self._get_top_movies(g)] for g in self._get_genres_ids()}
        seen_movies = set()

        movie_num = 0
        while movie_num < limit and cursors:
            min_genre, min_count, min_cursor = min(cursors.values(), key=operator.itemgetter(1))

            try:
                logging.debug("get_top_movies() - fetching movie {}".format(movie_num))
                movie = min_cursor.next()
                while movie["tmdb_id"] in seen_movies:
                    movie = min_cursor.next()

                yield movie
                movie_num += 1
                seen_movies.add(movie["tmdb_id"])

                for genre in movie["genres_ids"]:
                    cursor = cursors.get(genre, None)
                    if cursor:
                        cursor[1] += 1
            except StopIteration:
                del cursors[min_genre]


    def _get_crew(self, credits, job):
        results = []
        for crew in credits["crew"]:
            if crew["job"] == job:
                results.append(crew["name"])
        return results

    def get_movie(self, movie_id, size="w92", image_width=None):
        """Get the movie details.

        :param movie_id: this can either be a TMDB or an IMDB id.
        """
        logging.debug("get_movie({}, {})".format(movie_id, size))
        if size not in self.poster_sizes:
            raise Exception("Poster size {} is not in {}.".format(size, self.poster_sizes))

        params = {
          "append_to_response": "credits,keywords"
        }
        movie = json.loads(self._request("/movie/{}".format(movie_id), params))
        poster = StringIO(self._request_image("/{}{}".format(size, movie["poster_path"])))

        actors = movie["credits"]["cast"]
        actor1 = actors[0]["name"] if len(actors) > 0 else None
        actor2 = actors[1]["name"] if len(actors) > 1 else None

        directors = self._get_crew(movie["credits"], "Director")
        director = directors[0] if directors else None

        producers = self._get_crew(movie["credits"], "Producer")
        producer1 = producers[0] if producers else None

        writers = self._get_crew(movie["credits"], "Screenplay")
        writer1 = writers[0] if writers else None

        if image_width is None:
            image_width = int(size[1:])
        image_height = int(1.5*image_width)
        transformed_poster = transform.resize(imread(poster), (image_height, image_width, 3))

        return {
          "imdb_id": movie["imdb_id"],
          "tmdb_id": movie["id"],
          "actor1": actor1,
          "actor2": actor2,
          "director": director,
          "producer1": producer1,
          "writer1": writer1,
          "keywords": ",".join([k["name"] for k in movie["keywords"]["keywords"]]),
          "adult": movie["adult"],
          "budget": movie["budget"],
          "genres": [g["name"] for g in movie["genres"]],
          "genres_ids": [g["id"] for g in movie["genres"]],
          "language": movie["original_language"],
          "overview": movie["overview"],
          "poster": transformed_poster,
          "release_date": movie["release_date"],
          "revenue": movie["revenue"],
          "runtime": movie["runtime"],
          "tagline": movie["tagline"],
          "title": movie["title"],
          "popularity": movie["popularity"]
        }

    def _get_genres_ids(self):
        genres = json.loads(self._request("/genre/movie/list"))["genres"]
        return [g["id"] for g in genres]

    def get_genres(self):
        """Get the list of genres defined on TMDB"""
        genres = json.loads(self._request("/genre/movie/list"))["genres"]
        return [g["name"] for g in genres]
