import requests
from config import USER_AGENT, DEFAULT_TIMEOUT

def build_session():
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s
