import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from collections import defaultdict

def accuracy(model, df_pivot, reference_column='correct_answer'):
    return df_pivot.apply(lambda row: row[model] == row[reference_column], axis=1).mean()

if __name__ == "__main__":

            
    df_helm_model_pairs =pd.read_csv('./data/helm/model_correlations.csv')
    df_helm_model_pairs = pd.concat([df_helm_model_pairs, df_helm_model_pairs.rename(columns={'model1': 'model2', 'model2': 'model1'})])

    df_helm_accuracy = pd.read_csv('./data/helm/model_accuracy.csv', index_col=0)
    df_helm_accuracy = df_helm_accuracy.sort_values(by='model_accuracy')
    helm_models = df_helm_accuracy['model'].to_list()

    dict_error_agreement_helm = defaultdict(lambda: {})

    for index, row in df_helm_model_pairs.iterrows():
        dict_error_agreement_helm[row['model1']][row['model2']] = row['agreement_rate_when_both_wrong']

    for model in helm_models:
        dict_error_agreement_helm[model][model] = 1

    heatmap_data_helm = pd.DataFrame([[dict_error_agreement_helm[m1][m2] for m2 in helm_models] for m1 in helm_models[::-1]], index=helm_models[::-1], columns=helm_models)

    plt.figure(figsize=(50, 50))
    heatmap = sns.heatmap(heatmap_data_helm, cmap="viridis", vmax=0.8, vmin=0.33, cbar_kws={'label': 'Error Agreement Rate'})
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=60)
    cbar.ax.set_ylabel('Error Agreement Rate', fontsize=60)
    plt.xlabel("Model (Sorted by Accuracy)", fontsize=60)
    plt.ylabel("Model (Sorted by Accuracy)", fontsize=60)
    plt.title('Correlated Errors (HELM)', fontsize=90)
    plt.tight_layout()
    plt.savefig('./final_figures/heatmap_helm_71.png')
    


    df = pd.read_csv('./data/helm/all_mmlu_data_limitedcols.csv')
    df['question'] = df['subject'] + ' ' + df['id'] + ' ' +df['correct_answer_letter']

    df_pivot = df.pivot(index='question', columns='model', values='predicted_letter')
    df_pivot.reset_index(inplace=True)

    df_pivot['correct_answer'] = df_pivot['question'].apply(lambda x: x.split(' ')[-1])
    df_pivot.head()

    all_models = df['model'].unique()

    models_dict = {}
    for model in all_models:
        models_dict[model] = {}
        models_dict[model]['accuracy'] = accuracy(model, df_pivot)
        models_dict[model]['model_family'] = model.split('/')[0]

    all_model_families = [models_dict[model]['model_family'] for model in all_models]
    all_model_families = list(set(all_model_families))

    families_dict = {}
    for family in all_model_families:
        families_dict[family] = []

    for model in all_models:
        families_dict[models_dict[model]['model_family']].append(model)


    top_models = []

    accuracies = [accuracy(model, df_pivot) for model in all_models]
    n_families = len(all_model_families)
    n_cols = 5
    n_rows = (n_families + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()

    # compute accuracies of top model in each family, to use for sorting of plots
    top_accuracies = []
    for family in all_model_families:
        family_models = families_dict[family]
        family_accuracies = [models_dict[model]['accuracy'] for model in family_models]
        top_accuracies.append(np.max(family_accuracies))

    for idx, family in enumerate(np.array(all_model_families)[np.argsort(-np.array(top_accuracies))]):
        ax = axes[idx]
        family_models = families_dict[family]
        family_accuracies = [models_dict[model]['accuracy'] for model in family_models]
        top_model = family_models[np.argmax(family_accuracies)]
        accuracy_top_model = [accuracy(model, df_pivot,top_model) for model in all_models]
        top_models.append(top_model)

        family_models_idx = [i for i, model in enumerate(all_models) if models_dict[model]['model_family'] == family]
        not_family_models_idx = [i for i, model in enumerate(all_models) if models_dict[model]['model_family'] != family]
        ax.scatter(np.array(accuracies)[not_family_models_idx], np.array(accuracy_top_model)[not_family_models_idx], color='blue', alpha=0.75)
        ax.scatter(np.array(accuracies)[family_models_idx], np.array(accuracy_top_model)[family_models_idx], color='red', alpha=0.75)
        ax.plot([0.25, 1.02], [0.25, 1.02], 'k--')
        ax.plot([models_dict[top_model]['accuracy'], models_dict[top_model]['accuracy']], [0.25, 1.02], color='red')
        ax.set_xlim(0.25, 1.02)
        ax.set_ylim(0.25, 1.02)
        ax.set_title(f'{family}:\n{top_model}', fontsize=12)

    for idx in range(n_families, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('final_figures/llm_judge_helm.png', dpi=300)


    top_models = []

    accuracies = [accuracy(model, df_pivot) for model in all_models]
    n_families = len(all_model_families)
    n_cols = 5
    n_rows = (n_families + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3*n_rows))
    axes = axes.flatten()

    # compute accuracies of top model in each family, to use for sorting of plots
    top_accuracies = []
    for family in all_model_families:
        family_models = families_dict[family]
        family_accuracies = [models_dict[model]['accuracy'] for model in family_models]
        top_accuracies.append(np.max(family_accuracies))

    inflations_helm = []

    for idx, family in enumerate(np.array(all_model_families)[np.argsort(-np.array(top_accuracies))]):
        ax = axes[idx]
        family_models = families_dict[family]
        family_accuracies = [models_dict[model]['accuracy'] for model in family_models]
        top_model = family_models[np.argmax(family_accuracies)]
        accuracy_top_model = [accuracy(model, df_pivot,top_model) for model in all_models]
        top_models.append(top_model)

        family_models_idx = [i for i, model in enumerate(all_models) if models_dict[model]['model_family'] == family and model != top_model]
        not_family_models_idx = [i for i, model in enumerate(all_models) if models_dict[model]['model_family'] != family]

        sorted_accuracies = sorted(accuracies, reverse=True)
        ranks = [sorted_accuracies.index(accuracy) + 1 for accuracy in accuracies]
        sorted_accuracy_top_model = sorted(accuracy_top_model, reverse=True)
        ranks_top_model = [sorted_accuracy_top_model.index(accuracy) + 1 for accuracy in accuracy_top_model]

        inflations_helm.append(np.array(ranks_top_model)[family_models_idx] - np.array(ranks)[family_models_idx])

        ymin_not_family = np.min(np.array(accuracy_top_model)[not_family_models_idx] - np.array(accuracies)[not_family_models_idx]) if len(not_family_models_idx) > 0 else 0
        ymax_not_family = np.max(np.array(accuracy_top_model)[not_family_models_idx] - np.array(accuracies)[not_family_models_idx]) if len(not_family_models_idx) > 0 else 0
        ymin_family = np.min(np.array(accuracy_top_model)[family_models_idx] - np.array(accuracies)[family_models_idx]) if len(family_models_idx) > 0 else 0
        ymax_family = np.max(np.array(accuracy_top_model)[family_models_idx] - np.array(accuracies)[family_models_idx]) if len(family_models_idx) > 0 else 0
        ymin = np.min([ymin_not_family, ymin_family])
        ymax = np.max([ymax_not_family, ymax_family])

        ax.scatter(np.array(accuracies)[not_family_models_idx], np.array(accuracy_top_model)[not_family_models_idx] - np.array(accuracies)[not_family_models_idx], color='blue', alpha=0.75)
        ax.scatter(np.array(accuracies)[family_models_idx], np.array(accuracy_top_model)[family_models_idx] - np.array(accuracies)[family_models_idx], color='red', alpha=0.75)
        ax.plot([0.5, 1.02], [0, 0], 'k--')
        ax.plot([models_dict[top_model]['accuracy'], models_dict[top_model]['accuracy']], [ymin-0.01, ymax+0.01], color='red')
        ax.set_xlim(0.5, 1.02)
        ax.set_ylim(ymin-0.01, ymax+0.01)
        ax.set_xlabel('true accuracy', fontsize=12)
        ax.set_ylabel('accuracy inflation', fontsize=12)
        ax.set_title(f'{family}:\n{top_model}', fontsize=12)

    for idx in range(n_families, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('final_figures/llm_judge_helm_residuals.png', dpi=300)
