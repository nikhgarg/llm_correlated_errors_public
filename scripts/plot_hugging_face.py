import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
sns.set_style("white")

def accuracy(model, df_pivot, reference_column='correct_answer'):
    return df_pivot.apply(lambda row: row[model] == row[reference_column], axis=1).mean()

def get_accuracy_by_model(df, comp_col='correct_answer'):
    accuracies = {}
    for col in df.columns:
        accuracies[col] = (df[comp_col] == df[col]).mean()
    return accuracies
            
        

if __name__ == "__main__":

    relevant_models = []
    with open("./data/hugging_face/relevant_models.txt", "r") as f:
        for line in f:
            relevant_models.append(line.strip())

    df_model_pairs = pd.read_csv("./data/hugging_face/hf_featured_models.csv")
    hugging_face_accuracy = pd.read_csv("./data/hugging_face/hf_model_accuracy.csv", index_col=0)
    hugging_face_accuracy = hugging_face_accuracy.sort_values(by='0')
    hugging_face_accuracy = hugging_face_accuracy[hugging_face_accuracy.index.isin(relevant_models)]
    relevant_models = hugging_face_accuracy.index.to_list()
    df_model_pairs = pd.concat([df_model_pairs, df_model_pairs.rename(columns={'model1': 'model2', 'model2': 'model1'})]) # both directions
    df_model_pairs = df_model_pairs[df_model_pairs['model1'].isin(relevant_models) & df_model_pairs['model2'].isin(relevant_models)]

    df_models = pd.read_csv("./data/hugging_face/hf_model_accuracy.csv", index_col=0)
    df_models = df_models.sort_values(by='0')
    models = df_models.index.to_list()
    dict_error_agreement = defaultdict(lambda: {})

    for index, row in df_model_pairs.iterrows():
        dict_error_agreement[row['model1']][row['model2']] = row['agreement_rate_when_both_wrong']

    for model in relevant_models:
        dict_error_agreement[model][model] = 1
    
    heatmap_data = pd.DataFrame([[dict_error_agreement[m1][m2] for m2 in relevant_models] for m1 in relevant_models[::-1]], index=relevant_models[::-1], columns=relevant_models)

    plt.figure(figsize=(50, 50))
    heatmap = sns.heatmap(heatmap_data, cmap="viridis", vmax=0.5, cbar_kws={'label': 'Error Agreement Rate'})
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=60)
    cbar.ax.set_ylabel('Error Agreement Rate', fontsize=60)
    plt.xlabel("Model (Sorted by Accuracy)", fontsize=60)
    plt.ylabel("Model (Sorted by Accuracy)", fontsize=60)
    plt.title('Correlated Errors (HuggingFace)', fontsize=90)
    plt.tight_layout()
    plt.savefig('./final_figures/heatmap_hf_349.png')

    model_to_location_df = pd.read_csv('./data/hugging_face/model_to_location.csv', index_col=0)
    model_to_location_df = model_to_location_df[model_to_location_df.index.isin(relevant_models)]
    all_model_answers = pd.DataFrame()

    for model_name, row in model_to_location_df.iterrows():
        location = row['Location']
        tmp_df = pd.read_csv(location, index_col = 0)
        tmp_df.rename({'predicted_answer': model_name}, axis = 1, inplace=True)
        if all_model_answers.empty:
            all_model_answers = tmp_df[['doc_hash', 'correct_answer', model_name]]
        else:
            tmp_df = tmp_df[['doc_hash', model_name]]

            all_model_answers = pd.merge(all_model_answers, tmp_df, on='doc_hash')
    

    hf_model_info = pd.read_csv('./data/hugging_face/hugging_face.csv')
    hf_model_info = hf_model_info[hf_model_info['name'].isin(relevant_models)]
    model_to_architecture = dict(zip(hf_model_info['name'], hf_model_info['architecture']))

    all_models = relevant_models

    df_pivot = all_model_answers

    models_dict = {}
    for model in all_models:
        models_dict[model] = {}
        models_dict[model]['accuracy'] = accuracy(model, df_pivot)
        models_dict[model]['model_family'] = model_to_architecture[model]

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
        print(f"Family: {family} - {len(family_models)} \n")
        family_accuracies = [models_dict[model]['accuracy'] for model in family_models]
        top_accuracies.append(np.max(family_accuracies))

    inflations_hf = []

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

        inflations_hf.append(np.array(ranks_top_model)[family_models_idx] - np.array(ranks)[family_models_idx])

        ymin_not_family = np.min(np.array(accuracy_top_model)[not_family_models_idx] - np.array(accuracies)[not_family_models_idx]) if len(not_family_models_idx) > 0 else 0
        ymax_not_family = np.max(np.array(accuracy_top_model)[not_family_models_idx] - np.array(accuracies)[not_family_models_idx]) if len(not_family_models_idx) > 0 else 0
        ymin_family = np.min(np.array(accuracy_top_model)[family_models_idx] - np.array(accuracies)[family_models_idx]) if len(family_models_idx) > 0 else 0
        ymax_family = np.max(np.array(accuracy_top_model)[family_models_idx] - np.array(accuracies)[family_models_idx]) if len(family_models_idx) > 0 else 0
        ymin = np.min([ymin_not_family, ymin_family])
        ymax = np.max([ymax_not_family, ymax_family])

        ax.scatter(np.array(accuracies)[not_family_models_idx], np.array(accuracy_top_model)[not_family_models_idx] - np.array(accuracies)[not_family_models_idx], color='blue', alpha=0.75)
        ax.scatter(np.array(accuracies)[family_models_idx], np.array(accuracy_top_model)[family_models_idx] - np.array(accuracies)[family_models_idx], color='red', alpha=0.75)
        ax.plot([0.25, 1.02], [0, 0], 'k--')
        ax.plot([models_dict[top_model]['accuracy'], models_dict[top_model]['accuracy']], [ymin-0.01, ymax+0.01], color='red')
        ax.set_xlim(0.25, 1.02)
        ax.set_ylim(ymin-0.01, ymax+0.01)
        ax.set_xlabel('true accuracy', fontsize=12)
        ax.set_ylabel('accuracy inflation', fontsize=12)
        ax.set_title(f'{family}:\n{top_model}', fontsize=12)

    for idx in range(n_families, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('final_figures/llm_judge_hf_residuals.png', dpi=300)

    top_models = []

    accuracies = [accuracy(model, df_pivot) for model in all_models]
    n_families = len(all_model_families)
    n_cols = 5  # You can adjust this
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

    # Hide empty subplots if any
    for idx in range(n_families, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig('final_figures/llm_judge_hf.png', dpi=300)
    
    