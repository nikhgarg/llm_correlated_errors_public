import re, random
import pandas as pd
import numpy as np
import json, hashlib
from model_info import model_configs
import constants
import boto3


def parse_json(output):
    json_match = re.search(r"(\{.*?\})", output, re.DOTALL)

    if json_match:
        json_part = json_match.group(1).strip()
        try:
            json_data = json.loads(json_part)
            error_message = None
        except json.JSONDecodeError as e:
            error_message = f"JSON decoding error: {e}"
            fallback_match = re.search(r":\s*(\d+)", output)

            if fallback_match:
                extracted_integer = int(
                    fallback_match.group(1)
                )  # Extract the first integer found
                json_data = {"Score": extracted_integer}
            else:
                json_data = None
    else:
        json_data = None
        error_message = "No JSON structure found."

    return json_data, error_message


def hash_id(x):
    return hashlib.sha256(str(x).encode()).hexdigest()


def add_noise(df, filtered_columns):
    noise_scores_df = df.copy()
    for col in filtered_columns:
        noise = np.random.normal(loc=0, scale=1, size=len(df))
        noise_scores_df.loc[:, col] = (
            noise_scores_df[col].clip(lower=0, upper=10) + noise
        )
    return noise_scores_df


def run_api(modelId, model_configuration, prompt):
    """
    Calls an LLM model endpoint on AWS Bedrock with the specified configuration and prompt.

    Handles different model APIs by adjusting the request payload and parsing the response
    according to the model's requirements. Supports Titan, Nova, Mistral, Jamba, and other models.

    Args:
        modelId (str): The model identifier string (e.g., 'titan', 'nova', 'mistral', etc.).
        model_configuration (dict): Model-specific configuration parameters.
        prompt (str): The input prompt to send to the model.

    Returns:
        tuple:
            response_text (str): The raw text output from the model.
            model_json_response (dict or None): Parsed JSON object from the model output, or None if parsing fails.

    Raises:
        Exception: If the expected keyword is not found in the model response.
    """
    #     session = boto3.Session(profile_name='sso-elliot')
    session = boto3.Session(profile_name="sso-elliot", region_name="us-west-2")
    bedrock = session.client(service_name="bedrock-runtime")

    model = model_configs[modelId]
    config = {}
    if "titan" in modelId:
        config["textGenerationConfig"] = model_configuration.copy()
        config["inputText"] = prompt
        response = bedrock.invoke_model(
            body=json.dumps(config),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())["results"][0]
    elif "nova" in modelId:
        config["inferenceConfig"] = model_configuration.copy()
        config["messages"] = [{"role": "user", "content": [{"text": prompt}]}]
        response = bedrock.invoke_model(
            body=json.dumps(config),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        response_body = (
            json.loads(response.get("body").read())
            .get("output")
            .get("message")
            .get("content")[0]
        )
    elif modelId == "mistral.mistral-large-2407-v1:0":
        config = model_configuration.copy()
        config["prompt"] = prompt
        response = bedrock.invoke_model(
            body=json.dumps(config),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read()).get("choices")[0][
            "message"
        ]
    elif modelId == "mistral.mistral-large-2402-v1:0":
        config = model_configuration.copy()
        config["prompt"] = prompt
        response = bedrock.invoke_model(
            body=json.dumps(config),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        #         print(json.loads(response.get('body').read()))
        response_body = json.loads(response.get("body").read()).get("outputs")[0]
    elif modelId in [
        "mistral.mistral-7b-instruct-v0:2",
        "mistral.mixtral-8x7b-instruct-v0:1",
    ]:
        config = model_configuration.copy()
        config["prompt"] = prompt
        response = bedrock.invoke_model(
            body=json.dumps(config),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read()).get("outputs")[0]
    elif "jamba" in modelId:
        config = model_configuration.copy()
        config["messages"] = [{"role": "user", "content": prompt}]
        response = bedrock.invoke_model(
            body=json.dumps(config),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read()).get("choices")[0][
            "message"
        ]
    else:
        config = model_configuration.copy()
        config["prompt"] = prompt
        response = bedrock.invoke_model(
            body=json.dumps(config),
            modelId=modelId,
            accept="application/json",
            contentType="application/json",
        )
        response_body = json.loads(response.get("body").read())
    #     print(response_body)
    if model["keyword"] not in response_body:
        #         print(resume_job[1]['index'], response_body)
        raise Exception("Something went wrong")
    response_text = response_body.get(model["keyword"])
    model_json_response = parse_json(response_text)[0]
    return response_text, model_json_response
