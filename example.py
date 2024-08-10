import re, io, os, sys, ast, ssl, base64, sqlite_utils, logging
from datetime import datetime as DT, timezone as TZ, timedelta as TD

import pgpy, qrcode, vcfpy, vobject

async def function():
  info = 