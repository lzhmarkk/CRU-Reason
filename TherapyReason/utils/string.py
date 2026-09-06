import re


def parse_string(string):
    return re.sub(r"\d\.", "", string) \
        .replace("*", "") \
        .strip(": \n#")
