import requests
import configparser

cfg = configparser.ConfigParser()
cfg.read('api.ini')
apollo_api_key = cfg['common']['APOLLO_API_KEY'].split('=')
print(apollo_api_key)

def build_apollo_request(org_domain):
  url = "https://api.apollo.io/v1/mixed_people/search"
  data = {
    'api_key': apollo_api_key,
    'q_organization_domains': org_domain,
    'page': 1,
    'person_titles': ['data scientist manager', 'data scientist', 'project manager']
  }

  apollo_headers = {
    'Cache-Control': "no-cache",
    'Content-Type': 'application/json'
  }

  with requests.request("POST", url, headers=apollo_headers, json=data) as response:
    if response.status_code == 200 or not response.text:
      return response.text, json.loads(response.text)
    else:
      print(response.status_code)
      raise Exception(f'Failed to retrieve employee information for {org_domain}. {response.reason}')

p, people = build_apollo_request('jnj.com')
print(people['people'])