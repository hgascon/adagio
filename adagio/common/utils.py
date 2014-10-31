

from hashlib import sha256


def get_sha256(filename):
    """ Return sha256 of the file in the input path. """
    f = open(filename)
    s = sha256()
    s.update(f.read())
    digest = s.hexdigest()
    f.close()
    return digest
