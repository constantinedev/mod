import re, io, os, sys, ast, ssl, csv, json, requests
from tqdm import tqdm
from urllib.parse import urlparse
from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem
software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value, SoftwareName.SAFARI.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.MAC.value,]
user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=200)
user_agent = user_agent_rotator.get_random_user_agent()

session = requests.session()
session.proxies = {}
session.proxies['http'] = 'socks5h://localhost:9050'
session.proxies['https'] = 'socks5h://localhost:9050'

async def dwner(dwn_url, dwn_path, methods):
	filename = os.path.basename(urlparse(dwn_url).path)
	output_path = os.path.expanduser(dwn_path + filename)
	os.makedirs(os.path.dirname(output_path), exist_ok=True)
	if methods == "tor":
		response = session.get(dwn_url, stream=True)
	elif methods == "nor":
		response = requests.get(dwn_url, stream=True)
	total_size = int(response.headers.get('content-length', 0))
	block_size = 1024
	
	tqdm_bar = tqdm(total=total_size, unit="%", unit_scale=True, desc="Downloading")
	with open(dwn_path+filename, 'wb') as file:
		for data in response.iter_content(block_size):
			if data:
				tqdm_bar.update(len(data))
				file.write(data)
	tqdm_bar.close()
	print(f'Download Complete => {dwn_path + filename}')
