import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
import argparse

def analyze_model_correlations(model1_answers, model2_answers, correct_answers):
    """
    Analyze correlations between two models' answers, accounting for correct answers.
    
    Parameters:
    model1_answers: list of answers for model 1
    model2_answers: list of answers for model 2
    correct_answers: list of correct answers
    
    Returns:
    Dictionary containing various correlation metrics
    """
    # Create contingency table of model predictions
    contingency = pd.crosstab(model1_answers, model2_answers)
    
    # Perform chi-square test
    chi2, p_value, _, _ = chi2_contingency(contingency)
    
    # Calculate agreement rate
    agreement_rate = (model1_answers == model2_answers).mean()
    
    # Calculate agreement rate when both are correct
    both_correct_mask = (model1_answers == correct_answers) & (model2_answers == correct_answers)
    both_correct_rate = both_correct_mask.mean()
    
    # Calculate agreement rate when at least one is wrong
    either_wrong_mask = ~both_correct_mask
    agreement_rate_when_either_wrong = ((model1_answers == model2_answers) & either_wrong_mask).sum() / either_wrong_mask.sum() if either_wrong_mask.sum() > 0 else 0
    
    
    # agreement rate when both are wrong
    both_wrong_mask = (model1_answers != correct_answers) & (model2_answers != correct_answers)
    agreement_rate_when_both_wrong = ((model1_answers == model2_answers) & both_wrong_mask).sum() / both_wrong_mask.sum() if both_wrong_mask.sum() > 0 else 0
    
    
    return {
        'chi_square': chi2,
        'p_value': p_value,
        'overall_agreement_rate': agreement_rate,
        'both_correct_rate': both_correct_rate,
        'agreement_rate_when_either_wrong': agreement_rate_when_either_wrong,
        'agreement_rate_when_both_wrong': agreement_rate_when_both_wrong,
        'contingency_table': contingency
    }


if __name__ == "__main__":
    # Parse command line arguments for baseline model selection
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, default=None,
                       help="Optional baseline model to use for accuracy comparison")
    args = parser.parse_args()

    # Load list of models to analyze from predefined file
    relevant_models = []
    with open("./data/hugging_face/relevant_models.txt", "r") as f:
        for line in f:
            relevant_models.append(line.strip())

    # Load model metadata and location mappings
    model_to_location_df = pd.read_csv("./data/hugging_face/model_to_location.csv", index_col=0)
    hugging_face_info = pd.read_csv("./data/hugging_face/hugging_face.csv", index_col=0)
    
    # Filter out models with errors and get list of valid models
    no_error_data = model_to_location_df[~(model_to_location_df['Location'] == "error")]
    model_list = no_error_data[no_error_data.index.isin(relevant_models)].reset_index().values.tolist()

    baseline_model = args.baseline_model

    # If baseline model is provided, use it to calculate adjusted accuracies
    if baseline_model is not None:
        # Load baseline model predictions as ground truth
        baseline_model_df = pd.read_csv(model_to_location_df.loc[baseline_model]['Location'], index_col=0)
        correct_answer_df = baseline_model_df[["doc_hash", "predicted_answer"]].rename(
            {"predicted_answer": "correct_answer"}, axis=1)

        # Calculate accuracy relative to baseline model for each model
        new_model_accuracy = {}
        for model_name, row in model_to_location_df.iterrows():
            location = row['Location']
            model_df = pd.read_csv(location, index_col=0).drop('correct_answer', axis=1)
            model_df_with_answers = pd.merge(model_df, correct_answer_df, on="doc_hash")
            new_model_accuracy[model_name] = (model_df_with_answers['correct_answer'] == 
                                            model_df_with_answers['predicted_answer']).mean()
        
        # Save baseline-adjusted model accuracies
        hugging_face_model_accuracy = pd.DataFrame.from_dict(new_model_accuracy, orient='index')
        hugging_face_model_accuracy.to_csv(
            f"./data/hugging_face/hf_model_accuracy_{baseline_model.replace('/', '_')}.csv")
    else:
        # Calculate raw accuracy for each model using their own correct answers
        correctness = {}
        for model, row in model_to_location_df.iterrows():
            location = row['Location']
            if location != "error":
                df = pd.read_csv(location, index_col=0)
                correctness[model] = (df['predicted_answer'] == df['correct_answer']).mean()

        # Save raw model accuracies
        hugging_face_model_accuracy = pd.DataFrame.from_dict(correctness, orient='index')
        hugging_face_model_accuracy.to_csv("./data/hugging_face/hf_model_accuracy.csv")
    
    # Filter to only include relevant models and remove duplicates
    hugging_face_model_accuracy = hugging_face_model_accuracy[
        hugging_face_model_accuracy.index.isin(relevant_models)].drop_duplicates()
        
    # Compare models pairwise to analyze correlations and agreement rates
    results_all = []

    for en, model_1_info in enumerate(model_list):
        model_1_name, model_1_location = model_1_info
        print(f"Processing model {en}: {model_1_name}")
        
        # Compare with all subsequent models
        for en2, model_2_info in enumerate(model_list[en+1:]):
            model_2_name, model_2_location = model_2_info
            
            # Alternate model order to balance comparisons
            if (en + en2) % 2 == 0:
                model_1_name_tmp, model_2_name_tmp = model_2_name, model_1_name
                model_1_location_tmp, model_2_location_tmp = model_2_location, model_1_location
            else:
                model_1_name_tmp, model_2_name_tmp = model_1_name, model_2_name
                model_1_location_tmp, model_2_location_tmp = model_1_location, model_2_location    

            if model_1_name_tmp != model_2_name_tmp:
                # Load predictions from both models
                model_1_df = pd.read_csv(model_1_location_tmp, index_col=0)
                model_2_df = pd.read_csv(model_2_location_tmp, index_col=0)
                
                # Calculate correlation metrics using either baseline or model's own correct answers
                if baseline_model is not None:
                    total_with_answers = pd.merge(total, correct_answer_df, on="doc_hash")
                    results = analyze_model_correlations(
                        total_with_answers["predicted_answer_x"],
                        total_with_answers["predicted_answer_y"], 
                        total_with_answers['correct_answer'])
                else:                
                    total = pd.merge(model_1_df, model_2_df, on="doc_hash")
                    results = analyze_model_correlations(
                        total["predicted_answer_x"], 
                        total["predicted_answer_y"], 
                        total['correct_answer_x'])
                
                # Add model names to results
                results['model1'] = model_1_name_tmp
                results['model2'] = model_2_name_tmp
                
                results_all.append(results)
                
    # Convert results to DataFrame and save
    hugging_face_similarity = pd.DataFrame(results_all)
    
    df_switched = hugging_face_similarity.copy()
    np.random.seed(123)  # For reproducibility
    switch_mask = np.random.rand(len(df_switched)) > 0.5
    df_switched.loc[switch_mask, ['model1', 'model2']] = df_switched.loc[switch_mask, ['model2', 'model1']].values
    hugging_face_similarity = df_switched
    
    
    if baseline_model is not None:
        hugging_face_similarity.to_csv(f"./data/hugging_face/hf_model_correlation_{baseline_model.replace('/', '_')}.csv")
    else:
        hugging_face_similarity.to_csv("./data/hugging_face/hf_model_correlation.csv")

    # adding features
    company_name = hugging_face_info.reset_index()['index'].str.split("/", expand=True)[0].astype(str)
    hugging_face_model_accuracy.columns = ["accuracy"]
    hugging_face_info['company'] = list(company_name.values)
    hugging_face_relevant = hugging_face_info[["name", "params_billions", "is_moe", "generation", "architecture", "company"]]
    hugging_face_relevant = hugging_face_relevant.drop_duplicates(subset=["name"])

    hf_limited_with_model_1 = pd.merge(hugging_face_similarity, hugging_face_relevant, left_on="model1", right_on="name", suffixes=("", "_model1"), how="left")
    hf_limited_with_model_2 = pd.merge(hf_limited_with_model_1, hugging_face_relevant, left_on="model2", right_on="name", suffixes=("", "_2"), how="left")

    # additional features 
    hf_limited_with_model_2["param_diff"] = np.abs(hf_limited_with_model_2["params_billions"] - hf_limited_with_model_2["params_billions_2"])
    hf_limited_with_model_2["same_architecture"] = hf_limited_with_model_2["architecture"] == hf_limited_with_model_2["architecture_2"]
    hf_limited_with_model_2["same_company"] = hf_limited_with_model_2["company"] == hf_limited_with_model_2["company_2"]

    df = pd.merge(hf_limited_with_model_2, hugging_face_model_accuracy, left_on = "model1", right_index=True)
    df = pd.merge(df, hugging_face_model_accuracy, left_on = "model2", right_index=True)

    df.rename(columns = {"accuracy_x": "accuracy_1", "accuracy_y": "accuracy_2"}, inplace = True)
    if baseline_model is not None:
        df.to_csv(f"./data/hugging_face/hf_featured_models_{baseline_model.replace('/', '_')}.csv")
    else:
        df.to_csv("./data/hugging_face/hf_featured_models.csv")



