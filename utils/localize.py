from enum import Enum
import json

class supportedLocals(Enum):
    en = 0
    cn = 1
    ja = 2

supportedLocals_fullname = (
    "English",
    "Chinese(Mandarin)",
    "Japanese"
)

supportedLocals_tts_name = (
    "en",
    "zh",
    "ja"
)

class LocalStrNotImplemented(Exception):
    pass

class LocalFileNotFound(Exception):
    pass
class localStrFactory():
    def __init__(self, lang_name:str):
        fp = open(f"./localization/{lang_name}.json", "r")
        self.dic = json.load(fp)


    def get_str(self, key:str) -> str:
        try:
            return self.dic[key]
        except KeyError as e:
            raise LocalStrNotImplemented(key)
