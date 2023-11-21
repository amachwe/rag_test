
import openai
import logging
import datetime
import csv
import time


openai.organization = "org-h2jlUJiwWckRWJMMyD6vTWsu"
openai.api_key = "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"

logging.basicConfig(filename="gpt_query_log.txt",filemode="a", level=logging.INFO)
TIMESTAMP = datetime.datetime.now()
logging.info(f"\n=============================\nSession: {TIMESTAMP}")


USE_GPT4 = True


def fetch_openai(query):
   
    GPT3 = "gpt-3.5-turbo"  
    GPT4 = "gpt-4-0613"
    GPT_VERSION = GPT3
    if USE_GPT4:
        GPT_VERSION = GPT4
  
    messages = [
          {
                "role":"user", "content": query
            }
    ]


    response = openai.ChatCompletion.create(
        model=GPT_VERSION,
        messages=messages,
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=[]
        )
    usage = response.usage
    run_data = [usage.completion_tokens,  usage.prompt_tokens, usage.total_tokens, response.model, query]
    return extract_message_content(response), run_data

def extract_message_content(response):
    logging.info(response)
    return response.choices[0].message.content




if __name__ == "__main__":

    query = "My friend played Zork late into the night, he purchased it using his new Lloyds Credit card, I wonder what the game is about and if I will qualify for the card?"

    response, data = fetch_openai(query)
    logging.info(data)
    logging.info(response)
    print(response)




    

   
        