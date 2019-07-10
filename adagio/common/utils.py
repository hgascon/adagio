

from hashlib import sha256


def get_sha256(filename):
    """ Return sha256 of the file in the input path. """
    with open(filename, 'rb') as f:
        bytes = f.read()
    s = sha256()
    s.update(bytes)
    digest = s.hexdigest()
    f.close()
    return digest
