import pandas as pd 
import argparse
import numpy as np
from scipy.stats import chi2_contingency
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
    # Convert answers to numeric (A=0, B=1, etc.)
    # answer_to_num = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'Other': 4}
    # model1_answers_num = [answer_to_num.get(answer, 4) for answer in model1_answers]
    # model2_answers_num = [answer_to_num.get(answer, 4) for answer in model2_answers]
    # correct_answers_num = [answer_to_num.get(answer, 4) for answer in correct_answers]
    
    # Calculate raw correlation between models
    # raw_corr = np.corrcoef(model1_answers_num, model2_answers_num)[0, 1]
    
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
    
    
    # # Calculate partial correlation controlling for correct answers
    # def partial_corr(x, y, z):
    #     xy_corr = np.corrcoef(x, y)[0, 1]
    #     xz_corr = np.corrcoef(x, z)[0, 1]
    #     yz_corr = np.corrcoef(y, z)[0, 1]
        
    #     # Calculate partial correlation
    #     partial = (xy_corr - xz_corr * yz_corr) / (np.sqrt(1 - xz_corr ** 2) * np.sqrt(1 - yz_corr ** 2))
    #     return partial
    
    # partial_correlation = partial_corr(model1_answers_num, model2_answers_num, correct_answers_num)
    
    return {
        # 'raw_correlation': raw_corr,
        # 'partial_correlation': partial_correlation,
        'chi_square': chi2,
        'p_value': p_value,
        'overall_agreement_rate': agreement_rate,
        'both_correct_rate': both_correct_rate,
        'agreement_rate_when_either_wrong': agreement_rate_when_either_wrong,
        'agreement_rate_when_both_wrong': agreement_rate_when_both_wrong,
        'contingency_table': contingency
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_model", type=str, required=False, default=None)
    args = parser.parse_args()


    baseline_model = args.baseline_model
    mmlu_data = pd.read_csv('./data/helm/all_mmlu_data_limitedcols.csv')
    print("Loaded MMLU data")
    if baseline_model:
        print(f"Adjusting for {baseline_model}")
        relevant_model = mmlu_data[mmlu_data['model'] == baseline_model][['subject', 'id', 'predicted_letter']]
        relevant_model.rename({"predicted_letter": "correct_answer_letter"}, axis = 1, inplace=True)


        new_dataframes = []
        for unique_model in mmlu_data['model'].unique():
            tmp_dataframe = mmlu_data[mmlu_data['model'] == unique_model]
            with_correct_answer = pd.merge(tmp_dataframe[['model', 'subject', 'id', 'predicted_letter']], relevant_model, on=['subject', 'id'])
            new_dataframes.append(with_correct_answer)
        
        adjusted_dataframe = pd.concat(new_dataframes)
        adjusted_dataframe['got_correct_answer'] = adjusted_dataframe['predicted_letter'] == adjusted_dataframe['correct_answer_letter']
        adjusted_dataframe.to_csv(f"./data/helm/mmlu_data_limitedcols_{baseline_model.replace('/', '_')}.csv")


        # Get the model accuracy for each model
        model_correctness = adjusted_dataframe[['model', 'got_correct_answer']].groupby('model').mean().reset_index()
    else:
        model_correctness = mmlu_data[['model', 'got_correct_answer']].groupby('model').mean().reset_index()
    model_correctness.rename({'got_correct_answer': "model_accuracy"}, axis = 1, inplace=True)

    if baseline_model is None:
        model_correctness.to_csv(f"./data/helm/model_accuracy.csv")
    else:
        model_correctness.to_csv(f"./data/helm/model_accuracy{baseline_model.replace('/', '_')}.csv")
    

    # do model correlations 

    unique_questions = mmlu_data.question.unique()
    # df.groupby('question').correct_answer_letter.apply(lambda x: len(set(x))).max() # should be 1, each question should have only one correct answer
    if baseline_model is None:
        correct_answers = mmlu_data.groupby('question').correct_answer_letter.first()
        correct_answers_list = np.array(correct_answers[unique_questions].tolist())
        # correct_answers_list
    else:
        correct_answers_list = adjusted_dataframe.groupby('question').correct_answer_letter.first()
        correct_answers_list = np.array(correct_answers_list[unique_questions].tolist())

    model_predictions = {}

    for dfmodel in mmlu_data.groupby('model'):
        model = dfmodel[0]
        modeldf = dfmodel[1]
        model_predictions[model] = np.array(modeldf.groupby('question').predicted_letter.first()[unique_questions].tolist())
    # model_predictions

    all_models = mmlu_data['model'].unique()
    #dataframe where col1 is model1, col2 is model2, and the rest of the columns are the results of the comparison
    results_all = []
    print(len(all_models))

    for en, model1 in enumerate(all_models):
        print(en, model1)
        for en2, model2 in enumerate(all_models[en+1:]):
            if model1 != model2:
                # results = analyze_model_correlations(df, model1, model2)
                results = analyze_model_correlations(model_predictions[model1], model_predictions[model2], correct_answers_list)
                
                #results is a dictionary with the results of the comparison
                results['model1'] = model1
                results['model2'] = model2
                
                results_all.append(results)
                #append the results to the dataframe
                # results_df = results_df.append(results, ignore_index=True)
        # if en > 1:
            # break
            
    # dataframe from list of dictionaries
    results_df = pd.DataFrame(results_all)

    df_switched = results_df.copy()
    np.random.seed(123)  # For reproducibility
    switch_mask = np.random.rand(len(df_switched)) > 0.5
    df_switched.loc[switch_mask, ['model1', 'model2']] = df_switched.loc[switch_mask, ['model2', 'model1']].values
    results_df = df_switched

    if baseline_model is None:
        results_df.to_csv(f"./data/helm/model_correlations.csv")
    else:
        results_df.to_csv(f"./data/helm/model_correlations_{baseline_model.replace('/', '_')}.csv")