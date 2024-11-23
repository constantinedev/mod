import asyncio
import json
from scrapfly import ScrapflyClient, ScrapeConfig

from random_user_agent.user_agent import UserAgent
from random_user_agent.params import SoftwareName, OperatingSystem

software_names = [SoftwareName.CHROME.value, SoftwareName.FIREFOX.value, SoftwareName.SAFARI.value]
operating_systems = [OperatingSystem.WINDOWS.value, OperatingSystem.LINUX.value, OperatingSystem.MAC.value]
user_agent_rotator = UserAgent(software_names=software_names, operating_systems=operating_systems, limit=200)
user_agent = user_agent_rotator.get_random_user_agent()

SCRAPFLY = ScrapflyClient(key="scp-live-f8239ca90d5f48a9a51bfd8e637863ad")
BASE_CONFIG = {
	# Instagram.com requires Anti Scraping Protection bypass feature:
	# for more: https://scrapfly.io/docs/scrape-api/anti-scraping-protection
	"asp": True,
	"country": "CA",
}
INSTAGRAM_APP_ID = "936619743392459"  # this is the public app id for instagram.com

async def scrape_user(username):
	"""Scrape instagram user's data"""
	profile_config = await SCRAPFLY.async_scrape(
		ScrapeConfig(
			url=f"https://i.instagram.com/api/v1/users/web_profile_info/?username={username}",
			headers={"x-ig-app-id": INSTAGRAM_APP_ID, "User-Agent": user_agent},
			**BASE_CONFIG,
		)
	)
	target_data = profile_config.content
	user_id = json.loads(target_data)['data']['user']['id']
	# print(json.dumps(json.loads(target_data), ensure_ascii=False, indent=2))
	print(f"TARGET PIN: {user_id}")

	followers_data = await SCRAPFLY.async_scrape(
		ScrapeConfig(
			url = f" https://i.instagram.com/api/v1/friendships/{user_id}/followers/",
			headers = {'x-ig-app-id': INSTAGRAM_APP_ID, "User-Agent": user_agent},
			**BASE_CONFIG,
		)
	)
	followers_info = followers_data.content
	print(json.dumps(followers_info, ensure_ascii=False, indent=2))
	
	# return json.dumps(data["data"]["user"], ensure_ascii=False, indent=2)


asyncio.run(scrape_user("hkgov_discipline"))
