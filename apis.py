import re, io, os, sys, ast, ssl, requests, sqlite_utils, pgpy, logging
from datetime import datetime as DT, timezone as TZ, timedelta as TD
from sqlite_utils.utils import sqlite
from flask import Response, make_response, request, jsonify

async def api_loader(version):
  if version == "v1":
    return await v1()
  elif version == "v2":
    return await v2()

async def v1():
  if request.method == "GET":
    jsonData = {"response": "Web GET success."}
    return jsonify(jsonData), 200
  elif request.method == "POST":
    jsonData = {"response": "Permission Denied"}
    return jsonify(jsonData), 404

async def v2():
  if request.method == "GET":
    jsonData = {"response": "Permission Denied"}
    return jsonify(jsonData), 404
  elif request.method == "POST":
    jsonData = {"response": "POST request success."}
    return jsonify(jsonData), 200