import re, io, os, sys, ast, json, sqlite_utils, asyncio
from datetime import datetime as DT, timezone as TZ, timedelta as TD
import qrcode
from qrcode.image.svg import SvgPathFillImage

async def qr_svg(data, classname, box_size):
  if box_size == "" or box_size is None:
    box_size = 20
  if classname == "" or classname is None:
    classname = "qr_image"
  qr = qrcode.QRCode(image_factory=SvgPathFillImage, box_size=int(box_size))
  qr.add_data(data)
  img = qr.make_image(attrib={'class': classname, "name": classname})
  svg = img.to_string(encoding="unicode")
  print(svg)
  return svg
	
asyncio.run(qr_svg("123", None, None))
