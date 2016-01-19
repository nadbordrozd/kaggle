from collections import MutableMapping
import sqlite3
import pickle
from functools import wraps
from decorator import decorator


class PersistentDict(MutableMapping):
    def __init__(self, dbpath, iterable=None, **kwargs):
        self.dbpath = dbpath
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'create table if not exists memo '
                '(key blob primary key not null, value blob not null)'
            )
        if iterable is not None:
            self.update(iterable)
        self.update(kwargs)

    def encode(self, obj):
        return pickle.dumps(obj)

    def decode(self, blob):
        return pickle.loads(blob)

    def get_connection(self):
        return sqlite3.connect(self.dbpath)

    def __getitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select value from memo where key=?',
                (key,)
            )
            value = cursor.fetchone()
        if value is None:
            raise KeyError(key)
        return self.decode(value[0])

    def __setitem__(self, key, value):
        key = self.encode(key)
        value = self.encode(value)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'insert or replace into memo values (?, ?)',
                (key, value)
            )

    def __delitem__(self, key):
        key = self.encode(key)
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo where key=?',
                (key,)
            )
            if cursor.fetchone()[0] == 0:
                raise KeyError(key)
            cursor.execute(
                'delete from memo where key=?',
                (key,)
            )

    def __iter__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select key from memo'
            )
            records = cursor.fetchall()
        for r in records:
            yield self.decode(r[0])

    def __len__(self):
        with self.get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute(
                'select count(*) from memo'
            )
            return cursor.fetchone()[0]


def memo(cache, key_fun=None):
    def internalFunc(f):
        @wraps(f)
        def wrapper(f, *args, **kwargs):
            if key_fun:
                key = key_fun(*args, **kwargs)
            elif kwargs:
                key = args, frozenset(kwargs.iteritems())
            else:
                key = args
            #temporary fix - need to evaluate key upfront otherwise it may
            #change while evaluating the function
            key = pickle.dumps(key)
            if key in cache:
                return cache[key]
            else:
                cache[key] = result = f(*args, **kwargs)
                return result
        return decorator(wrapper, f)
    return internalFunc