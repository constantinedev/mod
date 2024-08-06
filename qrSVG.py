import re, io, os, sys, ast, json, sqlite_utils, asyncio
from datetime import datetime as DT, timezone as TZ, timedelta as TD
import qrcode, qrcode.constants
from qrcode.image.svg import SvgPathFillImage

async def qr_svg(data, classname, box_size):
  if box_size == "" or box_size is None:
    box_size = 20
  
  qr = qrcode.QRCode(
    version=5, 
    box_size=int(box_size), 
    border=0, 
    error_correction=qrcode.constants.ERROR_CORRECT_L, 
    image_factory=SvgPathFillImage
  )
  qr.add_data(data)

  if classname == "" or classname is None:
    classname = "qr_image"
	
  img = qr.make_image(attrib={'class': classname})
  svg = img.to_string(encoding="unicode")
  print(svg)
  return svg

##########
### asyncio.run(qr_svg("123", None, None))
##########
### You can save with the `write` method: 
### with open('filname.svg', 'w') as f:
###   f.write(svg)
### f.close()
