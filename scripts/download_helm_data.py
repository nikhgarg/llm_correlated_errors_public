import pandas as pd
import os
import requests
import json
import glob 


mmlu_scenarios = [
    "mmlu_abstract_algebra",
    "mmlu_anatomy",
    "mmlu_college_chemistry",
    "mmlu_computer_security",
    "mmlu_econometrics",
    "mmlu_global_facts",
    "mmlu_jurisprudence",
    "mmlu_philosophy",
    "mmlu_professional_medicine",
    "mmlu_us_foreign_policy",
    "mmlu_astronomy",
    "mmlu_business_ethics",
    "mmlu_clinical_knowledge",
    "mmlu_college_biology",
    "mmlu_college_computer_science",
    "mmlu_college_mathematics",
    "mmlu_college_medicine",
    "mmlu_college_physics",
    "mmlu_conceptual_physics",
    "mmlu_electrical_engineering",
    "mmlu_elementary_mathematics",
    "mmlu_formal_logic",
    "mmlu_high_school_biology",
    "mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science",
    "mmlu_high_school_european_history",
    "mmlu_high_school_geography",
    "mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics",
    "mmlu_high_school_mathematics",
    "mmlu_high_school_microeconomics",
    "mmlu_high_school_physics",
    "mmlu_high_school_psychology",
    "mmlu_high_school_statistics",
    "mmlu_high_school_us_history",
    "mmlu_high_school_world_history",
    "mmlu_human_aging",
    "mmlu_human_sexuality",
    "mmlu_international_law",
    "mmlu_logical_fallacies",
    "mmlu_machine_learning",
    "mmlu_management",
    "mmlu_marketing",
    "mmlu_medical_genetics",
    "mmlu_miscellaneous",
    "mmlu_moral_disputes",
    "mmlu_moral_scenarios",
    "mmlu_nutrition",
    "mmlu_prehistory",
    "mmlu_professional_accounting",
    "mmlu_professional_law",
    "mmlu_professional_psychology",
    "mmlu_public_relations",
    "mmlu_security_studies",
    "mmlu_sociology",
    "mmlu_virology",
    "mmlu_world_religions"
]

models = [
"ai21/jamba-instruct"
,"ai21/jamba-1.5-mini"
,"ai21/jamba-1.5-large"
,"anthropic/claude-instant-1.2"
,"anthropic/claude-2.1"
,"anthropic/claude-3-haiku-20240307"
,"anthropic/claude-3-sonnet-20240229"
,"anthropic/claude-3-opus-20240229"
,"anthropic/claude-3-5-sonnet-20240620"
,"anthropic/claude-3-5-sonnet-20241022"
,"cohere/command-r"
,"cohere/command-r-plus"
,"databricks/dbrx-instruct"
,"deepseek-ai/deepseek-llm-67b-chat"
,"google/gemini-1.0-pro-001"
,"google/gemini-1.5-pro-001"
,"google/gemini-1.5-flash-001"
,"google/gemini-1.5-pro-preview-0409"
,"google/gemini-1.5-flash-preview-0514"
,"google/gemini-1.5-pro-002"
,"google/gemini-1.5-flash-002"
,"google/gemma-7b"
,"google/gemma-2-9b"
,"google/gemma-2-27b"
,"google/text-bison@001"
,"google/text-unicorn@001"
,"meta/llama-2-7b"
,"meta/llama-2-13b"
,"meta/llama-2-70b"
,"meta/llama-3-8b"
,"meta/llama-3-70b"
,"meta/llama-3.1-8b-instruct-turbo"
,"meta/llama-3.1-70b-instruct-turbo"
,"meta/llama-3.1-405b-instruct-turbo"
,"meta/llama-3.2-11b-vision-instruct-turbo"
,"meta/llama-3.2-90b-vision-instruct-turbo"
,"microsoft/phi-2"
,"microsoft/phi-3-small-8k-instruct"
,"microsoft/phi-3-medium-4k-instruct"
,"01-ai/yi-6b"
,"01-ai/yi-34b"
,"01-ai/yi-large-preview"
,"allenai/olmo-7b"
,"allenai/olmo-1.7-7b"
,"mistralai/mistral-7b-v0.1"
,"mistralai/mistral-7b-instruct-v0.3"
,"mistralai/mixtral-8x7b-32kseqlen"
,"mistralai/mixtral-8x22b"
,"mistralai/mistral-small-2402"
,"mistralai/mistral-large-2402"
,"mistralai/mistral-large-2407"
,"mistralai/open-mistral-nemo-2407"
,"openai/gpt-3.5-turbo-0613"
,"openai/gpt-3.5-turbo-0125"
,"openai/gpt-4-1106-preview"
,"openai/gpt-4-0613"
,"openai/gpt-4-turbo-2024-04-09"
,"openai/gpt-4o-2024-05-13"
,"openai/gpt-4o-2024-08-06"
,"openai/gpt-4o-mini-2024-07-18"
,"qwen/qwen1.5-7b"
,"qwen/qwen1.5-14b"
,"qwen/qwen1.5-32b"
,"qwen/qwen1.5-72b"
,"qwen/qwen1.5-110b-chat"
,"qwen/qwen2-72b-instruct"
,"qwen/qwen2.5-7b-instruct-turbo"
,"qwen/qwen2.5-72b-instruct-turbo"
,"snowflake/snowflake-arctic-instruct"
,"writer/palmyra-x-v3"
,"writer/palmyra-x-004"
]

runvs = ['v1.0.0', 'v1.1.0', 'v1.2.0', 'v1.3.0', 'v1.4.0', 'v1.5.0', 'v1.6.0', 'v1.7.0', 'v1.8.0', 'v1.9.0', 'v1.10.0', 'v1.11.0']

def get_url_from_scenario_model(scenario, model, runv = 'v1.0.0'):
    scenario_without_mmlu = scenario[5:]
    base_url = 'https://storage.googleapis.com/crfm-helm-public/mmlu/benchmark_output/runs/{runv}/mmlu:subject={scenario_without_mmlu},method=multiple_choice_joint,model={model},eval_split=test,groups={scenario}/{scenario_state}.json'
    url = base_url.format(runv = runv, scenario_without_mmlu = scenario_without_mmlu, model=model, scenario=scenario, scenario_state='scenario_state')
    return url


def get_json_and_save(model, scenario, runvs):

    # Create the subfolder if it doesn't exist
    subfolder = "prediction_data/"
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
        
    filename = f"{subfolder}{model}_{scenario}.json"

    # Skip if JSON filename exists
    if os.path.exists(filename):
        print(f"JSON file {filename} already exists. Skipping download.")
        return
    # Get the JSON data from the URL
    
    success = False
    for runv in runvs:
        try:
            url = get_url_from_scenario_model(model=model, scenario=scenario, runv = runv)
            response = requests.get(url)
            json_data = response.json()

            # Save the JSON data to a file
            with open(filename, "w") as file:
                json.dump(json_data, file)

            print(f"JSON data saved to {filename}")
            success = True
            break
            
        except Exception as e:
            # print(f"Error downloading JSON data from {url}: {e}")
            continue
    if not success:
        print(f"Error downloading JSON data for {model} and {scenario} from {url}")

def parse_request_states(json_data):
    """
    Convert request_states from JSON into a pandas DataFrame.
    Each request_state item becomes a row with relevant fields extracted.

    # # Example usage
    # with open(json_file, 'r') as f:
    #     data = json.load(f)

    # df = parse_request_states(data)
    # print("\nDataFrame Shape:", df.shape)
    # print("\nColumns:", df.columns.tolist())
    # print("\nSample row:")
    # print(df.iloc[0])
    """
    rows = []
    
    for state in json_data['request_states']:
        result = state['result']
        completion = result['completions'][0]
        
        # Find the correct answer text and its corresponding letter
        correct_answer_text = None
        for i, ref in enumerate(state['instance']['references']):
            if 'correct' in ref.get('tags', []):
                correct_answer_text = ref['output']['text']
                break
        
        # Map the correct answer text to its letter using output_mapping
        correct_answer_letter = None
        for letter, text in state['output_mapping'].items():
            if text == correct_answer_text:
                correct_answer_letter = letter
                break
        
        # Extract relevant data from each state
        row = {
            # Instance info
            'id': state['instance']['id'],
            'split': state['instance']['split'],
            'question': state['instance']['input']['text'],
            'correct_answer_text': correct_answer_text,
            'correct_answer_letter': correct_answer_letter,
            'predicted_letter': completion['text'].strip(),
            'got_correct_answer': correct_answer_letter == completion['text'].strip(),
            
            # Request info
            'model': state['request']['model'],
            'temperature': state['request']['temperature'],
            'max_tokens': state['request']['max_tokens'],
            'top_p': state['request']['top_p'],
            'presence_penalty': state['request']['presence_penalty'],
            'frequency_penalty': state['request']['frequency_penalty'],
            
            # Result info
            'success': result['success'],
            'cached': result['cached'],
            # 'request_time': result['request_time'],
            'completion_text': completion['text'].strip(),
            'completion_logprob': completion['logprob'],
            # 'finish_reason': completion['finish_reason']['reason'],
        }
        
        # Add token-level information if available
        if 'tokens' in completion:
            tokens = completion['tokens']
            row['token_text'] = [t['text'] for t in tokens]
            row['token_logprobs'] = [t['logprob'] for t in tokens]
        
        # Add reference answers and their tags
        for i, ref in enumerate(state['instance']['references']):
            row[f'reference_{i+1}'] = ref['output']['text']
            row[f'reference_{i+1}_correct'] = 'correct' in ref.get('tags', [])
        
        # Add the output mapping (answer choices)
        for key, value in state['output_mapping'].items():
            row[f'choice_{key}'] = value
            
        rows.append(row)
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    # for scenario in mmlu_scenarios:
    #     for en, model in enumerate(models):
    #         get_json_and_save(model=model.replace('/', '_'), scenario=scenario, runvs = runvs)

    predictionfiles = glob.glob("prediction_data/*.json")
    dfs = []
    for file in predictionfiles:
        with open(file, 'r') as f:
            data = json.load(f)
        subject = file.split('_mmlu_')[1].split('.json')[0]
        df = parse_request_states(data)
        df['subject'] = subject
        dfs.append(df)
    dfall = pd.concat(dfs)
    dfall['question'] = dfall['question'].str.replace('\n', ' ').replace('"', '').replace("'", '').replace('`', '')
    dfall[['model','question','subject','id', 'correct_answer_letter', 'predicted_letter','got_correct_answer']].to_csv(
    './data/helm/all_mmlu_data_limitedcols.csv', index=False)

    unique_models = dfall['model'].unique()
    model_overview = pd.DataFrame(unique_models, columns = ['model string'])
    model_overview[['company', 'model_name']] = model_overview['model string'].str.split('/', expand=True)
    model_overview['size'] = model_overview['model string'].str.extract('(\d+b[a-z]*)')
    model_overview.to_csv('./data/helm/model_overview.csv', index=False)