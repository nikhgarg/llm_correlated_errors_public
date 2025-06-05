import boto3, re, json, random, hashlib, math, time, glob
import pandas as pd
import numpy as np
from constants import *
import matplotlib.pyplot as plt
import seaborn as sns
import deferred_acceptance as da
import scripts.setup_data as setup_data
from itertools import combinations


def add_noise(df, filtered_columns, use_hand=False):
    """
    Adds Gaussian noise (mean = 0, standard deviation = 1) to specified columns in a DataFrame.
    Optionally includes the 'Hand_firm_rate_comb' column if use_hand is True.
    Also clips the values to be between 0 and 10 after adding noise.

    Args:
        df (pd.DataFrame): Input DataFrame.
        filtered_columns (list): List of columns to which noise will be added.
        use_hand (bool): If True, also add noise to 'Hand_firm_rate_comb'.

    Returns:
        pd.DataFrame: DataFrame with noise added to specified columns.
    """
    if use_hand:
        df.reset_index(drop=True, inplace=True)
        filtered_columns = filtered_columns + ["Hand_firm_rate_comb"]
    noise_scores_df = df.copy()
    for col in filtered_columns:
        noise = np.random.normal(loc=0, scale=1, size=len(df))
        noise_scores_df.loc[:, col] = (
            noise_scores_df[col].clip(lower=0, upper=10) + noise
        )
    return noise_scores_df


def generate_random_applicant_pref(s_df, app_models):
    """
    Generates random applicant preference rankings for each model and resume.

    Args:
        s_df (pd.DataFrame): DataFrame containing 'Job_index' and 'Resume_index'.
        app_models (list): List of applicant model column names.

    Returns:
        pd.DataFrame: DataFrame with random rankings for each model and resume.
    """
    job_indices = s_df["Job_index"].unique()
    resume_indices = s_df["Resume_index"].unique()
    df = pd.DataFrame({"Job_index": job_indices})
    C = len(job_indices)
    new_columns = {}
    for model in app_models:
        for i, resume_index in enumerate(resume_indices):
            a = list(range(1, C + 1))
            random.shuffle(a)
            new_columns[f"{model}_r{i}"] = a
    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)
    return df


def generate_llm_applicant_pref(s_df, app_models):
    """
    Generates applicant preference rankings based on LLM model scores.

    Args:
        s_df (pd.DataFrame): DataFrame containing model scores.
        app_models (list): List of applicant model column names.

    Returns:
        pd.DataFrame: DataFrame with LLM-based rankings for each model and resume.
    """
    job_indices = s_df["Job_index"].unique()
    resume_indices = s_df["Resume_index"].unique()
    df = pd.DataFrame({"Job_index": job_indices})
    for model in app_models:
        for i, resume_index in enumerate(resume_indices):
            sample = s_df[s_df["Resume_index"] == resume_index]
            merged_df = pd.merge(df, sample[["Job_index", model]], on="Job_index")
            df[f"{model}_r{i}"] = (
                merged_df[model].rank(method="dense", ascending=False).astype("Int64")
            )
    return df


def generate_random_firm_pref(s_df, firm_models):
    """
    Generates random firm preference rankings for each model and job.

    Args:
        s_df (pd.DataFrame): DataFrame containing 'Job_index' and 'Resume_index'.
        firm_models (list): List of firm model column names.

    Returns:
        pd.DataFrame: DataFrame with random rankings for each model and job.
    """
    job_indices = s_df["Job_index"].unique()
    resume_indices = s_df["Resume_index"].unique()
    df = pd.DataFrame({"Resume_index": resume_indices})
    N = len(resume_indices)
    new_columns = {}
    for model in firm_models:
        for i, job_index in enumerate(job_indices):
            a = list(range(1, N + 1))
            random.shuffle(a)
            new_columns[f"{model}_j{i}"] = a
    new_columns_df = pd.DataFrame(new_columns)
    df = pd.concat([df, new_columns_df], axis=1)
    return df


def generate_llm_firm_pref(s_df, firm_models):
    """
    Generates firm preference rankings based on LLM model scores.

    Args:
        s_df (pd.DataFrame): DataFrame containing model scores.
        firm_models (list): List of firm model column names.

    Returns:
        pd.DataFrame: DataFrame with LLM-based rankings for each model and job.
    """
    job_indices = s_df["Job_index"].unique()
    resume_indices = s_df["Resume_index"].unique()
    df = pd.DataFrame({"Resume_index": resume_indices})
    for model in firm_models:
        for i, job_index in enumerate(job_indices):
            sample = s_df[s_df["Job_index"] == job_index]
            merged_df = pd.merge(df, sample[["Resume_index", model]], on="Resume_index")
            df[f"{model}_j{i}"] = (
                merged_df[model].rank(method="dense", ascending=False).astype("Int64")
            )
    return df


def generate_hand_firm_pref(s_df):
    """
    Generates firm preference rankings based on hand-labeled scores.

    Args:
        s_df (pd.DataFrame): DataFrame containing 'Hand_firm_rate_comb' column.

    Returns:
        pd.DataFrame: DataFrame with hand-labeled rankings for each job.
    """
    job_indices = s_df["Job_index"].unique()
    resume_indices = s_df["Resume_index"].unique()
    df = pd.DataFrame({"Resume_index": resume_indices})
    model = "Hand_firm_rate_comb"
    for i, job_index in enumerate(job_indices):
        sample = s_df[s_df["Job_index"] == job_index]
        merged_df = pd.merge(df, sample[["Resume_index", model]], on="Resume_index")
        df[f"{model}_j{i}"] = (
            merged_df[model].rank(method="dense", ascending=False).astype("Int64")
        )
    return df


def convert_df_tolist_by_cols(df, cols):
    res = []
    for column in cols:
        li = df[column].tolist()
        res.append(li)
    return res


def calculate_threshold_probability(df, group_columns, threshold):
    """
    Calculates the probability that values in each row exceed a given quantile threshold.

    Args:
        df (pd.DataFrame): Input DataFrame.
        group_columns (list): Columns to consider.
        threshold (float): Quantile threshold (e.g., 0.25 for top 25%).

    Returns:
        pd.Series: Probability for each row.
    """
    n = len(group_columns)
    thresholds = df[group_columns].quantile(threshold)

    def top_25_percent(row):
        failed = row > thresholds
        return failed.sum()

    probability = df[group_columns].apply(top_25_percent, axis=1)
    return probability / n


def make_unique_column_names(df):
    new_columns = []
    for idx, col in enumerate(df.columns):
        new_columns.append(f"{col}_{idx}")
    df.columns = new_columns


def get_sys_excl_rate(f_rank_df, firm_cols):
    """
    Calculates the systemic exclusion rate for firm rankings.

    Args:
        f_rank_df (pd.DataFrame): DataFrame of firm rankings.
        firm_cols (list): Columns to use for calculation.

    Returns:
        float: Systemic exclusion rate.
    """
    N = len(f_rank_df)
    sampled_df = f_rank_df[firm_cols]
    make_unique_column_names(sampled_df)
    threshold_probs = calculate_threshold_probability(
        sampled_df, sampled_df.columns, 0.25
    )
    systemic_failure_rate = (threshold_probs == 1.00).sum() / N
    return systemic_failure_rate


def split_into_k_sets(n, k):
    base_size = n // k
    remainder = n % k
    sets = [base_size + 1 if i < remainder else base_size for i in range(k)]

    return sets


def analyze_num_llm_homogeneity_rate(df, firm_models, filtered_columns):
    """
    Analyzes the failure rate as a function of the number of LLMs used for firm ranking.

    Args:
        df (pd.DataFrame): Input DataFrame.
        firm_models (list): List of firm model column names.
        filtered_columns (list): Columns to which noise will be added.

    Returns:
        pd.DataFrame: DataFrame with number of LLMs and corresponding failure rates.
    """
    num_applicants = len(df["Resume_index"].unique())
    C = len(df["Job_index"].unique())
    rows = []
    for i in range(1, len(firm_models) + 1):
        print(i)
        num_comb = math.comb(len(firm_models), i)
        total = num_comb
        distribution = split_into_k_sets(C, i)
        for _ in range(20):
            job_idx = random.sample(list(df["Job_index"].unique()), 1)[0]
            if num_comb > 100:
                samples = random.sample(list(combinations(firm_models, i)), 100)
                total = 100
            else:
                samples = combinations(firm_models, i)
            rates = 0
            for models in samples:
                t_df = df[df["Job_index"] == job_idx]
                t_df = add_noise(t_df, filtered_columns)
                order = []
                for model, item in zip(models, distribution):
                    order += [model] * item
                random.shuffle(order)

                sampled_df = t_df[order]
                rank_df = sampled_df.rank(method="dense", ascending=False).astype(
                    "Int64"
                )
                make_unique_column_names(rank_df)
                threshold_probs = calculate_threshold_probability(
                    rank_df, rank_df.columns, 0.25
                )
                systemic_failure_rate = (threshold_probs == 1.00).sum() / num_applicants
                rates += systemic_failure_rate
            rows.append({"num_llms": i, "fail_rate": rates / total})

    return pd.DataFrame(rows)


def perform_experiment(
    a_rank_df, f_rank_df, firm_cols, app_cols, firm_caps, diff_access=False
):
    """
    Runs a single matching experiment using applicant and firm preferences.

    Args:
        a_rank_df (pd.DataFrame): Applicant ranking DataFrame.
        f_rank_df (pd.DataFrame): Firm ranking DataFrame.
        firm_cols (list): Columns for firm preferences.
        app_cols (list): Columns for applicant preferences.
        firm_caps (list): Capacity for each firm.
        diff_access (bool or int): If diff_access = 1, applicants apply to top k choices;
        if 2, they apply to a random subset of k firms.

    Returns:
        tuple: (applicant_prefs, firm_prefs, applicant_matches, firm_matches, num_apps)
    """
    applicant_vals = convert_df_tolist_by_cols(a_rank_df, app_cols)
    applicant_prefs = da.prefs_from_values(applicant_vals)
    num_apps = []
    if diff_access:
        new_applicant_prefs = []
        for prefs in applicant_prefs:
            k = np.random.randint(1, len(a_rank_df) + 1)
            if diff_access == 1:
                new_applicant_prefs.append(prefs[:k])
            else:
                new_applicant_prefs.append(random.sample(prefs, k))
            num_apps.append(k)
        applicant_prefs = new_applicant_prefs

    firm_vals = convert_df_tolist_by_cols(f_rank_df, firm_cols)
    firm_prefs = da.prefs_from_values(firm_vals)
    applicant_matches, firm_matches = da.get_match(
        applicant_prefs, firm_prefs, firm_caps
    )
    return applicant_prefs, firm_prefs, applicant_matches, firm_matches, num_apps


def run_firm_experiments(
    scores_df,
    firm_models,
    filtered_columns,
    a_rank_df,
    app_cols,
    f_rank_df,
    with_hand_df,
    diff,
    human,
    firm_hand_vals=None,
    human_dic=None,
    diff_dic=None,
    job=None,
):
    """
    Runs a suite of firm matching experiments under different LLM and preference assignment scenarios.

    Args:
        scores_df (pd.DataFrame): Main scores DataFrame.
        firm_models (list): List of firm model column names.
        filtered_columns (list): Columns to which noise will be added.
        a_rank_df (pd.DataFrame): Applicant ranking DataFrame.
        app_cols (list): Columns for applicant preferences.
        f_rank_df (pd.DataFrame): Firm ranking DataFrame.
        with_hand_df (pd.DataFrame): DataFrame with hand-labeled scores.
        diff (bool or int): If True or int, applies differential access logic.
        human (bool): If True, uses human hand-labeled scores.
        firm_hand_vals (list, optional): Hand-labeled firm values.
        human_dic (dict, optional): Dictionary for storing human results.
        diff_dic (dict, optional): Dictionary for storing diff results.
        job (int, optional): Job index for experiment.

    Returns:
        dict or tuple: Results of the experiments, format depends on scenario.
    """
    vals = {}
    N = len(f_rank_df)
    C = len(a_rank_df)
    if diff:
        buckets = range(1, C + 1)
    elif human:
        buckets = np.arange(0, 10, 1)
    firm_caps = [1 for c in range(C)]
    # all firm use same LLM
    average_rank_df = pd.DataFrame()
    for model in firm_models:
        firm_cols = (
            [f"{model}_j{job}" for _ in range(C)]
            if job
            else [col for col in f_rank_df.columns if model in col]
        )
        applicant_prefs, firm_prefs, applicant_matches, firm_matches, _ = (
            perform_experiment(
                a_rank_df, f_rank_df, firm_cols, app_cols, firm_caps, diff
            )
        )
        average_rank_df.loc["Average Firm Rank", f"r_{model}"] = np.sum(
            da.college_value_of_match(firm_prefs, firm_matches)
        ) / np.sum(firm_caps)
        average_rank_df.loc["Average Applicant Rank", f"r_{model}"] = np.sum(
            da.student_rank_of_match(applicant_prefs, applicant_matches)
        ) / np.sum(firm_caps)
        if diff:
            for b, bucket in enumerate(buckets):
                for i in range(N):
                    if len(applicant_prefs[i]) == bucket:
                        diff_dic["Firms use Same LLM"][b].append(
                            applicant_matches[i] > -1
                        )
        elif human:
            average_rank_df.loc["Average Human Label Score", f"r_{model}"] = np.sum(
                [firm_hand_vals[firm[0]] for firm in firm_matches]
            ) / np.sum(firm_caps)
            for b, bucket in enumerate(buckets):
                for applicant, value in enumerate(firm_hand_vals):
                    if value < bucket + 1 and value >= bucket:
                        human_dic["Firms use Same LLM"][b].append(
                            applicant_matches[applicant] > -1
                        )
    vals["Firms use Same LLM_Firm Rank"] = average_rank_df.mean(axis=1).iloc[0]
    vals["Firms use Same LLM_Applicant Rank"] = average_rank_df.mean(axis=1).iloc[1]
    if human:
        vals["Firms use Same LLM_Hand Label Score"] = average_rank_df.mean(axis=1).iloc[
            2
        ]
    ## Firms use LLM from same company
    company_keywords = ["Llama", "Mistral", "Nova", "Claude", "Gpt"]
    company_models = {
        company: [model for model in firm_models if company in model]
        for company in company_keywords
    }
    for company, models in company_models.items():
        #         sampled_models = random.sample(models, 2)
        if job:
            #             firm_cols = [f"{sampled_models[0]}_j{job}" for _ in range(C//2)] + [f"{sampled_models[1]}_j{job}" for _ in range(C//2)]
            firm_cols = [f"{random.sample(models,1)[0]}_j{job}" for _ in range(C)]
        else:
            firm_cols = []
            for j in range(C):
                #                 col = f"{random.sample(sampled_models,1)[0]}_j{j}"
                col = f"{random.sample(models,1)[0]}_j{j}"
                firm_cols.append(col)
        applicant_prefs, firm_prefs, applicant_matches, firm_matches, _ = (
            perform_experiment(
                a_rank_df, f_rank_df, firm_cols, app_cols, firm_caps, diff
            )
        )
        if diff:
            for b, bucket in enumerate(buckets):
                for i in range(N):
                    if len(applicant_prefs[i]) == bucket:
                        diff_dic[f"{company}"][b].append(applicant_matches[i] > -1)
        elif human:
            vals[f"{company}_Hand Label Score"] = np.sum(
                [firm_hand_vals[firm[0]] for firm in firm_matches]
            ) / np.sum(firm_caps)
            for b, bucket in enumerate(buckets):
                for applicant, value in enumerate(firm_hand_vals):
                    if value < bucket + 1 and value >= bucket:
                        human_dic[f"{company}"][b].append(
                            applicant_matches[applicant] > -1
                        )

        vals[f"{company}_Firm Rank"] = np.sum(
            da.college_value_of_match(firm_prefs, firm_matches)
        ) / np.sum(firm_caps)
        vals[f"{company}_Applicant Rank"] = np.sum(
            da.student_rank_of_match(applicant_prefs, applicant_matches)
        ) / np.sum(firm_caps)

    ## Firms use Latest Models
    latest_models = [
        "Llama3-3-70b_firm_rate_comb2",
        "Mistral-Large_firm_rate_comb2",
        "Nova-Pro_firm_rate_comb2",
        "Claude_3.5_Sonnet(20241022)_firm_rate_comb2",
        "Gpt-4o-mini_firm_rate_comb2",
    ]
    if job:
        #         firm_cols = [f"{latest_models[0]}_j{job}" for _ in range(C//2)] + [f"{latest_models[1]}_j{job}" for _ in range(C//2)]
        firm_cols = [f"{random.sample(latest_models,1)[0]}_j{job}" for _ in range(C)]
    else:
        firm_cols = []
        for j in range(C):
            col = f"{random.sample(latest_models,1)[0]}_j{j}"
            firm_cols.append(col)
    applicant_prefs, firm_prefs, applicant_matches, firm_matches, _ = (
        perform_experiment(a_rank_df, f_rank_df, firm_cols, app_cols, firm_caps, diff)
    )
    if diff:
        for b, bucket in enumerate(buckets):
            for i in range(N):
                if len(applicant_prefs[i]) == bucket:
                    diff_dic[f"Latest"][b].append(applicant_matches[i] > -1)
    elif human:
        vals[f"Latest_Hand Label Score"] = np.sum(
            [firm_hand_vals[firm[0]] for firm in firm_matches]
        ) / np.sum(firm_caps)
        for b, bucket in enumerate(buckets):
            for applicant, value in enumerate(firm_hand_vals):
                if value < bucket + 1 and value >= bucket:
                    human_dic["Latest"][b].append(applicant_matches[applicant] > -1)

    vals[f"Latest_Firm Rank"] = np.sum(
        da.college_value_of_match(firm_prefs, firm_matches)
    ) / np.sum(firm_caps)
    vals[f"Latest_Applicant Rank"] = np.sum(
        da.student_rank_of_match(applicant_prefs, applicant_matches)
    ) / np.sum(firm_caps)

    ## Each Firm Uses Randomly Chosen LLM
    sampled_models = random.sample(firm_models, 5)
    if job:
        firm_cols = [f"{sampled_models[i % 5]}_j{job}" for i in range(C)]
    else:
        firm_cols = []
        for j in range(C):
            col = f"{random.sample(firm_models,1)[0]}_j{j}"
            firm_cols.append(col)

    applicant_prefs, firm_prefs, applicant_matches, firm_matches, _ = (
        perform_experiment(a_rank_df, f_rank_df, firm_cols, app_cols, firm_caps, diff)
    )
    #     print("RandomLLM: ", firm_cols, applicant_matches)
    if diff:
        for b, bucket in enumerate(buckets):
            for i in range(N):
                if len(applicant_prefs[i]) == bucket:
                    diff_dic[f"RandomLLM"][b].append(applicant_matches[i] > -1)
    elif human:
        vals[f"RandomLLM_Hand Label Score"] = np.sum(
            [firm_hand_vals[firm[0]] for firm in firm_matches]
        ) / np.sum(firm_caps)
        for b, bucket in enumerate(buckets):
            for applicant, value in enumerate(firm_hand_vals):
                if value < bucket + 1 and value >= bucket:
                    human_dic["RandomLLM"][b].append(applicant_matches[applicant] > -1)

    vals[f"RandomLLM_Firm Rank"] = np.sum(
        da.college_value_of_match(firm_prefs, firm_matches)
    ) / np.sum(firm_caps)
    vals[f"RandomLLM_Applicant Rank"] = np.sum(
        da.student_rank_of_match(applicant_prefs, applicant_matches)
    ) / np.sum(firm_caps)

    if human:
        random_f_rank_df = generate_random_firm_pref(
            add_noise(with_hand_df, filtered_columns), firm_models
        )
    else:
        random_f_rank_df = generate_random_firm_pref(
            add_noise(scores_df, filtered_columns), firm_models
        )
    sampled_cols = random.sample(list(random_f_rank_df.iloc[:, 1:].columns), 5)
    firm_cols = [sampled_cols[i % 5] for i in range(C)]
    applicant_prefs, firm_prefs, applicant_matches, firm_matches, _ = (
        perform_experiment(
            a_rank_df, random_f_rank_df, firm_cols, app_cols, firm_caps, diff
        )
    )
    if diff:
        for b, bucket in enumerate(buckets):
            for i in range(N):
                if len(applicant_prefs[i]) == bucket:
                    diff_dic[f"RandomPref"][b].append(applicant_matches[i] > -1)
    elif human:
        vals[f"RandomPref_Hand Label Score"] = np.sum(
            [firm_hand_vals[firm[0]] for firm in firm_matches]
        ) / np.sum(firm_caps)
        for b, bucket in enumerate(buckets):
            for applicant, value in enumerate(firm_hand_vals):
                if value < bucket + 1 and value >= bucket:
                    human_dic["RandomPref"][b].append(applicant_matches[applicant] > -1)
    vals[f"RandomPref_Firm Rank"] = np.sum(
        da.college_value_of_match(firm_prefs, firm_matches)
    ) / np.sum(firm_caps)
    vals[f"RandomPref_Applicant Rank"] = np.sum(
        da.student_rank_of_match(applicant_prefs, applicant_matches)
    ) / np.sum(firm_caps)

    if diff:
        return diff_dic
    elif human:
        return human_dic, vals
    else:
        return vals
