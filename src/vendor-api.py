import pandas as pd

import os
from datetime import datetime
from typing import Optional
import json
import openai
import cohere
import ai21
from transformers import AutoModel, pipeline

import joblib


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
        assert os.path.exists(
            directory
        ), f"Directory provided in `filename` does not exist: ({directory})"

    json_content = json.dumps(dictionary)
    with open(filename, "w") as f:
        f.write(json_content)
        return filename


PROMPT = "How is generative AI affecting the infrastrucutre machine learning developers need access to?"
run_name = "vendor-api-{}".format(datetime.now().strftime("%Y%m%dT%H%M"))

openai_client = openai.OpenAI(
    api_key=openai_api_key,
)

gpt35_completion = openai_client.chat.completions.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": PROMPT}]
)
gpt35_response = gpt35_completion.to_dict()
out = write_dict_to_file(
    gpt35_response, os.path.join("prompt_results", "openai", run_name, "response.json")
)
assert out != None, f"Unable to write OpenAI output"
