import re, io, os, sys, ast, json, sqlite_utils, requests, asyncio
from datetime import datetime as DT, timezone as TZ, timedelta as TD
from aiohttp_socks import ProxyType, ProxyConnector, ProxyConnectionError

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.UNIX.value, OperatingSystem.MAC.value]
hardware_types = [HardwareType.MOBILE.value, HardwareType.SERVER.value]
software_engines = [SoftwareEngine.GECKO.value, SoftwareEngine.WEBKIT.value, SoftwareEngine.BLINK.value]
software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value, SoftwareName.SAFARI.value, SoftwareName.ANDROID.value]
user_agent_rotator = UserAgent(software_names=software_names, software_engines=software_engines, hardware_types=hardware_types, operating_systems=operating_systems, limit=100)
user_agent = user_agent_rotator.get_random_user_agent()

session = requests.session()
session.proxies = {}
session.proxies['http'] = 'socks5h://127.0.0.1:9050'
session.proxies['https'] = 'socks5h://127.0.0.1:9050'

# headers = {'User-Agent': user_agent}
# payload = {}

async def gun_shell(url, method, type, headers, payload):
  if url == "" or url is None:
    return {"status":"Error","response":"URL was empty!"}
  if headers=={} or headers is None:
    headers = {'User-Agent': user_agent}
  if payload=={} or payload is None:
    payload={}

  if type == "nor":
    if method == "GET":
      req = requests.get(url, headers=headers, data=payload)
    elif method == "POST":
      req = requests.post(url, headers=headers, data=payload)
  elif type == "tor":
    if method == "GET":
      req = session.get(url, headers=headers, data=payload)
    elif method == "POST":
      req = session.post(url, headers=headers, data=payload)
  retData = {
    "status": req.status_code,
    "type": type,
    "method": method,
    "response": req.text
  }
  return retData
