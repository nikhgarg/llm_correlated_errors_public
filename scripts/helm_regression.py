import pandas as pd 
import numpy as np 
import seaborn as sns 

import statsmodels.formula.api as smf

def run_and_save_regression(y, xs, df):
    # Define the formula for the regression
    formula = f"{y} ~ {' + '.join(xs)}" #+ "+ company1 : company2"
    
    # Create the regression model
    model = smf.ols(formula=formula, data=df)

    # Fit the model to the data
    results = model.fit()

    # Print the summary of the regression results
    print(results.summary())
    
    # put it in latex format and then save it
    latex = results.summary().as_latex()
    with open(f'./regressions/helm/{y}_helm_regression.tex', 'w') as f:
        f.write(latex)

y_values = ['overall_agreement_rate', 'both_correct_rate',
    'agreement_rate_when_either_wrong', 'agreement_rate_when_both_wrong']

features = [
    'same_company',
    'accuracy_1',
    'accuracy_2',
]

if __name__ == "__main__":

    dfcorr = pd.read_csv('./data/helm/model_correlations.csv')
    dfmodels = pd.read_csv('./data/helm/model_overview.csv').rename(columns={'model string': 'model'})
    dfpreds = pd.read_csv('./data/helm/all_mmlu_data_limitedcols.csv')

    # get average helm accuracies 
    dfmodelaccuracies = dfpreds.groupby(
        'model')['got_correct_answer'].mean().reset_index().sort_values('got_correct_answer', ascending=False).rename(columns={'got_correct_answer': 'mean_accuracy'})
    df_merged = dfmodels.merge(dfmodelaccuracies, on='model', how='left')

    colstoprint = ['model', 'company', 'accuracy']
    df_mergedprint = df_merged.sort_values(['company','mean_accuracy'], ascending=False).reset_index(drop = True).rename(columns = {'mean_accuracy': 'accuracy'})
    tabletex = df_mergedprint[colstoprint].to_latex(longtable = True, multirow = True, index = True, caption = 'Models analyzed from Helm', float_format="{:.2f}".format, label = 'tab:helmmodels'
    )
    #print to file
    with open('final_figures/helmmodels.tex', 'w') as f:
        f.write(tabletex)


    df_merged_model1 = df_merged.rename(columns={x : x + '1' for x in df_merged.columns})
    dfcorr = dfcorr.merge(df_merged_model1[['model1', 'company1', 'mean_accuracy1']], on='model1', how='left')

    df_merged_model2 = df_merged.rename(columns={x : x + '2' for x in df_merged.columns})
    df = dfcorr.merge(df_merged_model2[['model2', 'company2', 'mean_accuracy2']], on='model2', how='left')

    df.rename(columns={'mean_accuracy1': 'accuracy_1', 'mean_accuracy2': 'accuracy_2'}, inplace=True)

    df['same_company'] = df['company1'] == df['company2']

    dfstandard = df.copy()

    features_to_standardize = [
        'accuracy_1', 'accuracy_2',
    ]

    #standardize the numeric columns 
    for col in features_to_standardize:
        if dfstandard[col].dtype == 'float64':
            dfstandard[col] = (dfstandard[col] - dfstandard[col].mean()) / dfstandard[col].std()

    for y in y_values:
        run_and_save_regression(y, features + ["accuracy_1*accuracy_2"], df=dfstandard)