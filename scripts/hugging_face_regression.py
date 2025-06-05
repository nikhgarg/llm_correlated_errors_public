import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf

def run_and_show_regression(y, xs, df, datasetname='hf'):
    # Define the formula for the regression
    formula = f"{y} ~ {' + '.join(xs)}" #+ "+ company1 : company2"
    
    print(df.shape)
    # Create the regression model
    model = smf.ols(formula=formula, data=df)

    # Fit the model to the data
    results = model.fit()

    # Print the summary of the regression results
    print(results.summary())
    
    latex = results.summary().as_latex()
    
    # remove text Notes: 
    # [1] Standard Errors assume that the covariance matrix of the errors is correctly specified. from the string
    latex = latex.replace('Notes: \\newline', '')
    latex = latex.replace('[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.', '')
    # strip newlines from end
    latex = latex.strip()
    
    with open(f'regressions/hugging_face/{y}_{datasetname}_regression.tex', 'w') as f:
        f.write(latex)

features_to_standardize = [
       'params_billions', 'params_billions_2', 'param_diff',
       'accuracy_1', 'accuracy_2',
       'params_billions_log', 'params_billions_log_2']


if __name__ == "__main__":
    df = pd.read_csv('./data/hugging_face/hf_featured_models.csv')
    print(df.shape)
    df['params_billions_log'] = np.log(df['params_billions'])
    df['params_billions_log_2'] = np.log(df['params_billions_2'])
    df.drop_duplicates(subset=["model1", "model2"], inplace=True)

    print(df.shape)
    y_values = ['overall_agreement_rate', 'both_correct_rate',
       'agreement_rate_when_either_wrong', 'agreement_rate_when_both_wrong']

    features = [
        'same_company',
        'same_architecture',
        'params_billions_log',
        'params_billions_log_2',
        'is_moe',
        'is_moe_2',
        'generation',
        'generation_2',
        'param_diff',
        'accuracy_1',
        'accuracy_2',
    ]

    col = 'agreement_rate_when_both_wrong'
    dfstandard = df.copy()


    #standardize the numeric columns 
    for col in features_to_standardize:
        if dfstandard[col].dtype == 'float64':
            dfstandard[col] = (dfstandard[col] - dfstandard[col].mean()) / dfstandard[col].std()

    for y in y_values:
        run_and_show_regression(y, features + ['accuracy_1*accuracy_2'], df=dfstandard)