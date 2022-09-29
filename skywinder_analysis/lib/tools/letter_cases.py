import re


def from_camel_to_lower(name):
    '''
    Copied from: http://stackoverflow.com/a/1176023/721964
    '''
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()
