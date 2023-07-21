import os
import openai
import requests
import json
import configparser
import re
import time
import datetime
import pandas as pd
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.memory import SimpleMemory
from langchain.prompts import PromptTemplate

cfg = configparser.ConfigParser()
cfg.read('api.ini')
openai_api_key = cfg['common']['OPENAI_API_KEY']
apollo_api_key = cfg['common']['APOLLO_API_KEY']
clearbit_api_key = cfg['common']['CLEARBIT_API_KEY']

llm = OpenAI(temperature=0.7, openai_api_key = openai_api_key)

sectors = ['healthcare', 'e-learning', 'food delivery', 'outdoor', 'gaming']

def find_companies_request(sector):
  message = '''
    Give me a list of 10 companies that are based in this {sector}. \n\n
    Include 5 companies that are primarily based in the sector and the rest as smaller companies and emerging startups. \n\n
    The format of your output should be a single line with each company's name separated by a comma and a space. \n\n
    Just write a list of the the companies with no bullets or numbering. \n\n
  '''

  prompt_template = PromptTemplate(input_variables=['sector'], template=message)
  initial_chain = LLMChain(llm=llm, prompt=prompt_template, output_key = 'initial')

  overall_chain = SequentialChain(
      chains=[initial_chain],
      input_variables = ['sector'],
      output_variables=['initial'],
      verbose=False,
  )

  return overall_chain({'sector': sector })['initial']

def find_company_url(company):
  company_formatted = company.replace(' ', '%20')
  url = f'https://autocomplete.clearbit.com/v1/companies/suggest?query=:{company_formatted}'
  with requests.request('GET', url) as response:
    if response.status_code == 200 and response.text != '[]':
      return list(json.loads(response.text))[0]['domain']
    else:
      raise Exception(f'Failed to retrieve company domain for {company}.')

def build_apollo_request(org_domain):
  url = "https://api.apollo.io/v1/mixed_people/search"
  data = {
    'api_key': apollo_api_key,
    'q_organization_domains': org_domain,
    'page': 1,
    'person_titles': ['data scientist manager', 'data scientist', 'project_manager', 'analytics manager', 'sales manager']
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

def build_chain(org_domain, personal):
  org_domain = org_domain.split('.')[0]
  # Chain 1: Generating initial email
  message = '''
    I represent a student data science consulting organization on UC Berkeley campus. \n\n
    We strive to promote data science in our community through educational bootcamps and industry-level data consulting. I would like to write emails to industry professionals in data science asking them if they are interested in partnerships between our organizations. 
    Specifically, our organization, Big Data at Berkeley would like to offer consulting services in data science and data engineering. \n\n
    We work on a range of projects like anomaly detection models, customer clustering, churn and growth analysis, data-driven revenue expansion strategies, etc. \n\n
    Help me write emails to these professionals, who are typically data scientists or data managers, pitching our organization as a perfect partner for a semester-long project.  
  '''
  template = "{message}\n\n"
  prompt_template = PromptTemplate(input_variables=['message'], template=template)
  initial_chain = LLMChain(llm=llm, prompt=prompt_template, output_key = 'initial')

  # Chain 2: Refine to be company-specific
  template = """
    You wrote this email for our club: \n\n
    {initial} \n\n
    Here is a company that we want to write the email to: \n\n
    {org_domain} \n\n
    Use information about a specific company such as their mission statements and innovations to tailor the prevous email to that company. \n\n
    Within the email, include a numbered list of specific examples of potential data science project collaborations that focus on addressing potential needs of the company. \n\n
    Some examples include customer churn and growth analysis, customer clustering and personalization, anomaly detection models, and data-driven revenue expansion strategy. \n\n
    In addition, make a numbered list of specific advantages our club can offer to the company. \n\n
    Make this email 400 words or more for now. 
  """
  prompt_template = PromptTemplate(input_variables=["initial", 'org_domain'], template=template)
  tailored_chain = LLMChain(llm=llm, prompt=prompt_template, output_key = 'final_1')

  # Chain 3: Workaround for max response tokens on ChatGPT 3.5
  template = """
    If you were cut off, please continue to finish the email from the previous prompt.
    If not, please return an empty line.
  """
  prompt_template = PromptTemplate(input_variables = [], template=template)
  continue_1_chain = LLMChain(llm=llm, prompt=prompt_template, output_key='continue_1')

  # Chain 4: adding a personal touch to the email.
  template = '''
  You wrote this email for the club:\n\n
  {final_1} + {continue_1} \n\n
  I also want to add a personal touch to the email to make it not sound too generic. \n\n
  Please incorporate this information into the email as well. \n\n
  {personal}
  '''
  prompt_template = PromptTemplate(input_variables = ['final_1', 'continue_1', 'personal'], template=template)
  personal_chain = LLMChain(llm=llm, prompt=prompt_template, output_key='final_2')

  template = """
    If you were cut off, please continue to finish the email from the previous prompt.
    If not, please return an empty line.
  """
  prompt_template = PromptTemplate(input_variables = [], template=template)
  continue_2_chain = LLMChain(llm=llm, prompt=prompt_template, output_key='continue_2')

  # Chain 5: Shortening email response to 200 words.
  template = """
    You wrote this email for the specific company in the previous prompt:
    {final_2} + {continue_2}
    Try limit the email to 250 words. \n\n
    Please keep the personal touch part of the email, as well as company-specific services our club can provide. \n\n
    Make the subject line as "Big Data at Berkeley: Partner with Us for Data Science Projects". \n\n
    In addition, replace the person being written to with [PERSON_NAME], and replace sender name with [YOUR NAME].
  """
  prompt_template = PromptTemplate(input_variables=['final_2', 'continue_2'], template=template)
  limit_chain = LLMChain(llm=llm, prompt=prompt_template, output_key = 'limit')

  #Chain 6: 
  template = """
    If you were cut off, please continue to finish the email from the previous prompt. \n
    If not, please return an empty line.
  """
  prompt_template = PromptTemplate(input_variables = [], template=template)
  continue_3_chain = LLMChain(llm=llm, prompt=prompt_template, output_key='continue_3')

  overall_chain = SequentialChain(
      chains=[initial_chain, tailored_chain, continue_1_chain, personal_chain, continue_2_chain, limit_chain, continue_3_chain],
      input_variables = ['message', 'org_domain', 'personal'],
      output_variables=['initial', 'final_2', 'continue_2', 'limit', 'continue_3'],
      verbose=False,
  )

  # Running all the chains on the user's question and displaying the final answer
  return overall_chain({'message': message, 'org_domain': org_domain, 'personal': personal})

def generate_csv(to_write, contactor):
  table = {
    'date': [],
    'name': [],
    'company': [],
    'email': [],
    'contactor': [],
    'title': []
  }
  today = datetime.datetime.now().strftime('%m/%d')
  for company in to_write:
    company = dict(company)
    if not company['people']:
      continue
    else:
      for person in company['people']:
        name = person.get('first_name', ' ') + ' ' + (person.get('last_name', ' ') or ' ')
        table['company'].append(company['company_name'])
        table['date'].append(today)
        table['name'].append(name)
        table['email'].append(person['email'])
        table['contactor'].append(contactor)
        table['title'].append(person['title'])
  pd.DataFrame.from_dict(table).to_csv('data/contacts.csv')
with open('data/sample_emails.json','r',encoding='utf-8') as f:

  to_write = json.load(f)


# to_write = []
# with open('data/personal.json', 'r', encoding='utf-8') as f:
#   personals = dict(json.load(f))

# for sector in sectors: 
#   print(sector)
#   pat = '[0-9]+\.'
#   companies = re.sub(pat, '', find_companies_request(sector)).replace('\n', '').lower().split(', ')[:5]
#   personal = personals.get(sector, '')
#   print(personals.get(sector, ''))
#   for company in companies:
#     try:
#       org_domain = find_company_url(company)
#       print(org_domain)
#       response = build_chain(org_domain, personal)
#       email = response['limit'] + response['continue_3']
#       p, people = build_apollo_request(org_domain)
#       people = [
#         {
#           'first_name': person['first_name'],
#           'last_name': person['last_name'],
#           'title': person['title'],
#           'linkedin': person['linkedin_url'],
#           'email': person['email']
#         } for person in people['people'] if person['email'] and person['email_status'] == 'verified'
#       ][:10]

#       full = {
#         'company_name': company,
#         'generated_email': email,
#         'people': people
#       }

#       to_write.append(full)

        
#     except Exception as e:
#       print(e)
#       continue

# with open('data/sample_emails.json', 'w+', encoding='utf-8') as f:
#   f.write(json.dumps(to_write, indent=4))

generate_csv(to_write, 'Matthew')