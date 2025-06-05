import requests 
import pandas as pd 
import json
import numpy as np
from scipy.stats import chi2_contingency
from huggingface_hub import HfFileSystem
import os 

def get_splits_url_from_textcode(textcode):
    """
    Retrieves the dataset split URL from the raw code block
    """
    splits = json.loads(textcode.split("\n")[3][9:].replace("'", '"'))
    hugging_face_url = (textcode.split("\n")[4]).split('"')[1]
    url_with_bbh = hugging_face_url + splits['latest']
    return url_with_bbh.replace("bbh_boolean_expressions", "mmlu_pro")


def get_actual_resp(resp):
    """
    The answers are given as a softmax 
    """
    highest_number = [i for i,val in enumerate(resp) if val[0][0] == max(resp, key = lambda x: float(x[0][0]))[0][0]]
    return highest_number[0]


if __name__ == "__main__":

    hugging_face_token = os.getenv("HF_TOKEN")
    print(f"Using token: {hugging_face_token}")
    fs = HfFileSystem(token=hugging_face_token)

    # grabbing the general model information from hugging face
    all_data = requests.get("https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted")
    all_data_json = all_data.json()

    overall_df = pd.DataFrame.from_dict(all_data_json)
    overall_df.set_index("id", inplace=True)

    final_df = None
    # overall df has nested json, so this explodes the information
    for col in overall_df.columns:
        tmp_df = pd.DataFrame.from_dict(overall_df[col].to_dict(), orient="index")
        if final_df is not None:
            final_df = pd.merge(final_df, tmp_df, left_index=True,right_index=True)
        else:
            final_df = tmp_df 

    # saving the interim data file
    final_df.to_csv("./data/hugging_face/hugging_face.csv")

    urls = pd.read_csv("./data/hugging_face/pandas_urls_2100.csv", index_col=0)

    relevant_models = []
    with open("./data/hugging_face/relevant_models.txt", "r") as f:
        for line in f:
            relevant_models.append(line.strip())

    # huggingface-cli login

    hugging_face_total = pd.merge(final_df, urls[["0", "1"]], left_on = "name", right_index=True)

    new_dataframe_dict = {}


    for _, row in hugging_face_total.sort_values("average_score", ascending=False).iterrows():
        model_name = row['name']
        if model_name in new_dataframe_dict:
            continue 
        if model_name not in relevant_models:
            continue 
        try:
            if "bbh_boolean_expressions" in row["0"]:  
                processed_df = pd.read_json(get_splits_url_from_textcode(row["0"]), lines=True)
            else:
                processed_df = pd.read_json(get_splits_url_from_textcode(row["1"]), lines=True)
            minimized = processed_df[["doc_hash", "acc"]]
            minimized['correct_answer'] = processed_df['doc'].apply(lambda x: x['answer_index'])
            minimized['predicted_answer'] = processed_df['resps'].apply(get_actual_resp)
            no_slash_model_name = model_name.replace("/", "_")
            file_path = f"./processed_data/{no_slash_model_name}.csv"
            minimized.to_csv(file_path)
            new_dataframe_dict[model_name] = file_path
            print(f"just saved to {file_path}")
        except:
            new_dataframe_dict[model_name] = "error"
        if len(new_dataframe_dict) > 500:
            break 


    model_to_location_df = pd.DataFrame.from_dict(new_dataframe_dict, orient="index")
    model_to_location_df.columns = ["Location"]
    # saving the model to location of the data in ./processed_data
    model_to_location_df.to_csv("./data/hugging_face/model_to_location.csv")

