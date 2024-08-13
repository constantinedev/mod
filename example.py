import re, io, os, sys, ast, ssl, base64, sqlite_utils, logging
from datetime import datetime as DT, timezone as TZ, timedelta as TD
from flask import request, make_response, Response, jsonify, redirect, url_for, abort, flash

import pgpy, qrcode, qrcode.img.svg, vobject
from modules.cmsmod import tokMaler, tokRecovery, pgpEnc, pgpDec, svgQRMaker, FX_2FA

mod = request.args.get('mod')
pag = request.args.get('pag')
json_data = request.get_json()
headerFrom = request.headers

async def api_loader(version):
  if version == "v1":
    if request.method == "GET":
      return await v1()
  if version == "v2":
    if request.method == "POST":
      token = headerFrom['token']
      for agent in sqlite_utils.Database['db/admin.db3']('users').rows_where("token == :tok", {'tok':token})
        return await v2()
      else:
        return jsonify('response': 'Permission Denied'), 200

async def v1():
  if mod == "encryptmsg":
    return await pgpEnc(json_data)

async def v2():
  if mod == "mkqr":
    return await svgQRMaker(json_data)
    
