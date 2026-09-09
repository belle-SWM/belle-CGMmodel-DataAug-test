"""
Author: SWM-Benjamin

"""


def get_username(uuid):
    assert isinstance(uuid, int), '\'uuid\' should be int !'
    uid = UUID_dict.get(uuid)
    return uid


def get_uuid(username):
    assert isinstance(username, str), '\'username\' should be string !'
    name = UUID_dict[username]
    return name


UUID_dict = {
    'Ray': 6,
    'Vincent': 7,
    'Jesse-7': 9,
    'penny': 30,
    'ElmerChen': 38,
    'Tracy': 45,
    'vin': 46,
    'dennis': 48,
    'Apple': 63,
    'Ariel': 67,
    'David Lee': 91,
    '林sharon': 139,
    '陳誼寧': 323,
    'Ming-Che': 378,
    'Jared Liu': 388,
    '李政達': 419,
    '歐巴桑': 434,
    'Jenny Tseng': 502,
    'Benjamin': 548,
    '江慶宗': 599,
    'Queena': 637,
    'Benjamin2': 726,
    'queena2': 727,
    'Lin': 816,
    'Fiona': 829
}

