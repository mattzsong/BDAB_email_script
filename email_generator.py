import os
import openai
import requests
import json
from langchain.chains import LLMChain, SequentialChain
from langchain.llms import OpenAI
from langchain.memory import SimpleMemory
from langchain.prompts import PromptTemplate
openai_api_key = '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'
apollo_api_key = 'yNBZvnsSb-OcjcWfPkvEkA'

llm = OpenAI(temperature=0.7, openai_api_key = openai_api_key)

apollo_headers = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/json'
}

companies=  ['doordash.com']

def build_apollo_request(org_domain):
  url = "https://api.apollo.io/v1/mixed_people/search"
  data = {}
  data['api_key'] = apollo_api_key
  data['q_organization_domains'] = [org_domain]
  data['person_titles'] = ['data science manager']

  apollo_headers = {
    'Cache-Control': 'no-cache',
    'Content-Type': 'application/json'
  }

  with requests.request("POST", url, headers=apollo_headers, json=data) as response:
    if response.status_code == 200:
      return response.text, json.loads(response.text)
    else:
      raise Exception(f'Failed to retrieve employee information for {org_domain}')

def build_chain(org_domain):
  org_domain = org_domain.split('.')[0]
  # Chain 1: Generating initial email
  message = '''
    I represent a student data science consulting organization on UC Berkeley campus. 
    We strive to promote data science in our community through educational bootcamps and industry-level data consulting. I would like to write emails to industry professionals in data science asking them if they are interested in partnerships between our organizations. 
    Specifically, our organization, Big Data at Berkeley would like to offer consulting services in data science and data engineering. 
    We work on a range of projects like anomaly detection models, customer clustering, churn and growth analysis, data-driven revenue expansion strategies, etc. 
    Help me write emails to these professionals, who are typically data scientists or data managers, pitching our organization as a perfect partner for a semester-long project.  
  '''
  template = """{message}\n\n"""
  prompt_template = PromptTemplate(input_variables=['message'], template=template)
  initial_chain = LLMChain(llm=llm, prompt=prompt_template, output_key = 'initial')

  # Chain 2: Refine to be company-specific
  template = """
    You wrote this email for our club:
    {initial}
    Here is a company that we want to write the email to:
    {org_domain} \n\n
    Use information about a specific company such as their mission statements and innovations to tailor the prevous email to that company.
    Within the email, include a numbered list of specific examples of potential data science project collaborations that focus on addressing potential needs of the company.
    Some examples include customer churn and growth analysis, customer clustering and personalization, anomaly detection models, and data-driven revenue expansion strategy.
    In addition, make a numbered list of specific advantages our club can offer to the company.
    Make this email 400 words or more for now.
  """
  prompt_template = PromptTemplate(input_variables=["initial", 'org_domain'], template=template)
  tailored_chain = LLMChain(llm=llm, prompt=prompt_template, output_key = 'final')

  # Chain 3: Workaround for max response tokens on ChatGPT 3.5
  template = """
    If you were cut off, please continue to finish the email from the previous prompt.
    If not, please return an empty line.
  """
  prompt_template = PromptTemplate(input_variables = [], template=template)
  continue_chain = LLMChain(llm=llm, prompt=prompt_template, output_key='continue')

  # Chain 4: Shortening email response to 200 words.
  template = """
    You wrote this email for the specific company in the previous prompt:
    {final} + {continue}
    While maintaing our qualifications and examples of the projects, try and limit the email to 200 words.
    In addition, replace the person being written to with [PERSON_NAME].
  """
  prompt_template = PromptTemplate(input_variables=['final', 'continue'], template=template)
  limit_chain = LLMChain(llm=llm, prompt=prompt_template, output_key = 'limit')
  overall_chain = SequentialChain(
      chains=[initial_chain, tailored_chain, continue_chain, limit_chain],
      input_variables = ['message', 'org_domain'],
      output_variables=['initial', 'final', 'continue', 'limit'],
      verbose=True,
  )

  # Running all the chains on the user's question and displaying the final answer
  return overall_chain({'message': message, 'org_domain':org_domain})

for org_domain in companies:
  try:
    email = build_chain(org_domain)['limit']
    p, people = build_apollo_request(org_domain)
    print(p)
    print(people['people'])
    people = [
      {
        'first_name': person['first_name'],
        'last_name': person['last_name'],
        'title': person['title'],
        'linkedin': person['linkedin_url'],
        'email': person['email']
      } for person in people if person['email_status'] == 'verified' and person['state'] == 'California'
    ][:10]

    with open('sample_emails.txt', 'w+', encoding='utf-8') as f:
      for person in people:
        to_write = f'{person["linkedin"]} \n{person["email"]} \n{email.replace("[PERSON_NAME]", first_name)}'
        person.write(to_write)
      
  except Exception as e:
    print(e)
    continue

