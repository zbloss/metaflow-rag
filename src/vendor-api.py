import pandas as pd

import os
from datetime import datetime
from typing import Optional
import json
import openai
import cohere
import ai21
from transformers import AutoModel, pipeline
from dotenv import load_dotenv
load_dotenv()

def write_dict_to_file(dictionary: dict, filename: str) -> Optional[str]:
    """
    Dumps `dictionary` to a JSON-object and
    writes the content to a file at `filename`.

    Arguments:
        dictionary : dict
            Dictionary to write to a file.
        filename : str
            The file you want to write `dictionary`
            to.
    Returns:
        filename : Optional[str]
            `filename` if the file was succesfully
            written. `None` if file was not written.
    """

    directory, _ = os.path.split(filename)
    if directory != "":

        os.makedirs(directory, exist_ok=True)

        assert os.path.exists(
            directory
        ), f"Directory provided in `filename` does not exist: ({directory})"

    json_content = json.dumps(dictionary)
    with open(filename, "w") as f:
        f.write(json_content)
        return filename
    

def call_open_ai(prompt: str, api_key: Optional[str] = None, model: str = 'gpt-3.5-turbo') -> dict:
    """
    Helper function to call the OpenAI ChatGPT endpoint
    for Chat Completion.

    Arguments: 
        prompt : str
            Prompt you want to send for Chat Completion.
        api_key : Optional[str]
            Optional API Key, defaults to `OPENAI_API_KEY`
        model : str
            Model you want to use from OpenAI, defaults to
            'gpt-3.5-turbo'.
    Returns
        api_response : dict
            Response from the ChatGPT API as a dictionary.
    """

    OPENAI_API_KEY: str = os.getenv('OPENAI_API_KEY')
    if api_key:
        OPENAI_API_KEY: str = api_key

    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

    print('Calling OpenAI...')
    gpt_completion = openai_client.chat.completions.create(
        model=model, 
        messages=[{"role": "user", "content": prompt}]
    )
    print('Done...')
    api_response: dict = gpt_completion.model_dump()
    return api_response

def call_cohere(prompt: str, api_key: Optional[str] = None, model: str = 'command') -> dict:
    """
    Helper function to call the Cohere endpoint
    for Text Generation.

    Arguments: 
        prompt : str
            Prompt you want to send for Text Generation.
        api_key : Optional[str]
            Optional API Key, defaults to `COHERE_API_KEY`
        model : str
            Model you want to use from Cohere, defaults to
            'command'.
    Returns
        api_response : dict
            Response from the Cohere API as a dictionary.
    """

    COHERE_API_KEY: str = os.getenv('COHERE_API_KEY')
    if api_key:
        COHERE_API_KEY: str = api_key

    cohere_client = cohere.Client(COHERE_API_KEY)

    print('Calling Cohere...')
    cohere_response = cohere_client.generate(prompt=prompt, model=model)
    cohere_response_data = cohere_response.data[0]
    api_response = {
        'id': cohere_response_data.id,
        'text': cohere_response_data.text,
        'likelihood': cohere_response_data.likelihood,
        'finish_reason': cohere_response_data.finish_reason,
    }
    print('Done...')
    return api_response


def call_ai21(prompt: str, api_key: Optional[str] = None, model: str = "j2-mid") -> dict:
    """
    Helper function to call the AI 21 endpoint
    for Text Generation.

    Arguments: 
        prompt : str
            Prompt you want to send for Text Generation.
        api_key : Optional[str]
            Optional API Key, defaults to `AI21_API_KEY`
        model : str
            Model you want to use from AI 21, defaults to
            'j2-mid'.
    Returns
        api_response : dict
            Response from the AI21 API as a dictionary.
    """

    AI21_API_KEY: str = os.getenv('AI21_API_KEY')
    if api_key:
        AI21_API_KEY: str = api_key

    ai21.api_key = AI21_API_KEY
    
    print('Calling AI 21...')
    jurassic2_completion = ai21.Completion.execute(
        model=model, 
        prompt=prompt,
        maxTokens=250
    )
    print('Done...')
    jurassic2_response = jurassic2_completion.completions
    api_response: dict = {
        'text': jurassic2_response[0]['data'].text.strip()
    }
    return api_response

def huggingface_generation(prompt: str, model_name: str = 'bigscience/bloom-560m'):
    """
    Helper function to generate text completion using
    the huggingface/transformers library.

    Arguments: 
        prompt : str
            Prompt you want to send for Text Generation.
        model_name : str
            Model you want to use from Huggingface, defaults to
            'bigscience/bloom-560m'.
    Returns
        response : dict
            Text generated as a dictionary.
    """
     
    generator = pipeline("text-generation", model=model_name, device_map="auto")
    generated_response: str = generator(prompt, do_sample=False, max_new_tokens=250)
    response = {
        'text': generated_response
    }
    return response


if __name__ == '__main__':

    PROMPT = "How is generative AI affecting the infrastrucutre machine learning developers need access to?"
    run_name = "vendor-api-{}".format(datetime.now().strftime("%Y%m%dT%H%M"))
    print(f'run_name: {run_name}')

    ### CALLING CHATGPT ###
    gpt35_response: dict = call_open_ai(PROMPT)
    out = write_dict_to_file(
        gpt35_response, os.path.join("prompt_results", "openai", run_name, "response.json")
    )
    assert out != None, f"Unable to write OpenAI output"

    ### CALLING COHERE ###
    cohere_response: dict = call_cohere(PROMPT)
    out = write_dict_to_file(
        cohere_response, os.path.join("prompt_results", "cohere", run_name, "response.json")
    )
    assert out != None, f"Unable to write Cohere output"
    
    ### CALLING AI 21 ###
    ai21_response: dict = call_ai21(PROMPT)
    out = write_dict_to_file(
        ai21_response, os.path.join("prompt_results", "ai21", run_name, "response.json")
    )
    assert out != None, f"Unable to write AI 21 output"

    ### USING HF/TRANSFORMERS ###

    hf_response: dict = huggingface_generation(PROMPT)
    out = write_dict_to_file(
        hf_response, os.path.join("prompt_results", "transformers", run_name, "response.json")
    )
    assert out != None, f"Unable to write Huggingface Transformers output"
