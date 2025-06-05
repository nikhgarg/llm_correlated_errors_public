import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

"""
You must first download the datasets from the following links:
- https://www.kaggle.com/datasets/asaniczka/upwork-job-postings-dataset-2024-50k-records --> store in data/upwork-jobs.csv
- https://github.com/florex/resume_corpus --> store in data/resume_samples.txt
- https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset --> store in data/Resume.csv
"""


def split_and_validate(row):
    parts = row.split(":::")
    if len(parts) == 3:
        return {"id": parts[0], "occupation": parts[1], "resume": parts[2]}
    return None


def load_raw_data():
    """
    Loads and processes raw job and resume data from specified CSV and TXT files.
    
    This function:
    1. Loads job postings and creates a 'Content' column combining title and description
    2. Loads resumes from two sources (txt and CSV files)
    3. Validates and concatenates resume data
    4. Filters resumes by length (between 5000 and 10000 characters)

    Returns:
        tuple: (resume_df, jobs_df)
            resume_df (pd.DataFrame): DataFrame containing processed resumes
            jobs_df (pd.DataFrame): DataFrame containing job postings with combined 'Content' column
    """

    jobs_df = pd.read_csv("../data/upwork-jobs.csv")
    jobs_df["Content"] = (
        "Job Title: " + jobs_df["title"] + "\nDescription: " + jobs_df["description"]
    )

    # load resume data from two different sources and concatenate
    f = open("../data/resume_samples.txt", "r", encoding="windows-1252")
    txt = f.read()
    data = [
        split_and_validate(row) for row in txt.split("\n") if split_and_validate(row)
    ]
    a = pd.read_csv("../data/Resume.csv")
    # Filter out invalid entries and remove duplicates
    data = [d for d in data if d is not None]
    b = pd.DataFrame(data).drop_duplicates(subset="resume")

    resume_df = pd.DataFrame(
        pd.concat([b["resume"], a["Resume_str"]]), columns=["Resume"]
    )
    resume_df = resume_df[
        (resume_df["Resume"].str.len() > 5000) & (resume_df["Resume"].str.len() < 10000)
    ].reset_index(drop=True)

    return resume_df, jobs_df


def generate_embeddings(resume_df, job_df):
    """
    Generates sentence embeddings for resumes and job descriptions using a pre-trained transformer model.

    Args:
        resume_df (pd.DataFrame): DataFrame containing a 'Resume' column.
        job_df (pd.DataFrame): DataFrame containing a 'Content' column.

    Returns:
        tuple: (resume_embeddings, job_embeddings)
            resume_embeddings (np.ndarray): Embeddings for resumes.
            job_embeddings (np.ndarray): Embeddings for job descriptions.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_embeddings = model.encode(list(resume_df["Resume"]))
    job_embeddings = model.encode(list(job_df["Content"]))

    return resume_embeddings, job_embeddings


def calculate_similarity(resume_embeddings, job_embeddings):
    """
    Calculates cosine similarity between clusters of resume and job embeddings for different cluster sizes.
    For each k (from 2 to 50, step 2), clusters resumes and jobs, computes centroid similarities,
    and prints the top 2 most similar cluster pairs.

    Args:
        resume_embeddings (np.ndarray): Embeddings for resumes.
        job_embeddings (np.ndarray): Embeddings for job descriptions.

    Returns:
        list: List of top similarity values for each k.
    """
    similarities = []
    for k in range(2, 52, 2):
        num_clusters = k
        kmeans_resumes = KMeans(n_clusters=num_clusters, random_state=0).fit(
            resume_embeddings
        )
        kmeans_jobs = KMeans(n_clusters=num_clusters, random_state=0).fit(
            job_embeddings
        )
        resume_centroids = kmeans_resumes.cluster_centers_
        job_centroids = kmeans_jobs.cluster_centers_
        similarity_matrix = cosine_similarity(resume_centroids, job_centroids)
        similarity_flat = similarity_matrix.flatten()
        top_2_indices = np.argsort(similarity_flat)[-20:][::-1]
        top_2_clusters = np.unravel_index(top_2_indices, similarity_matrix.shape)

        similarity = similarity_matrix[top_2_clusters[0][0], top_2_clusters[1][0]]
        similarities.append(similarity)
        print(f"k={k}")
        print("\nTop 2 most similar clusters:")
        for idx in range(2):
            print(
                f"Pair {idx+1}: Resume Cluster {top_2_clusters[0][idx]} and Job Cluster {top_2_clusters[1][idx]} with similarity {similarity_matrix[top_2_clusters[0][idx], top_2_clusters[1][idx]]}"
            )
            print(f"Similarity: {similarity}")
    return similarities


model_key = ["Llama", "Mistral", "Claude", "Nova", "Gpt"]


def filter_and_prepare_data(df):
    """
    Filters and prepares the input DataFrame for analysis:
    - Removes columns containing certain keywords.
    - Converts relevant columns to numeric, coercing errors to NaN.
    - Fills NA values in all columns except 'Hand_firm_rate_comb' with the column mean.
    - Creates a subset DataFrame containing only rows with hand-labeled firm ratings.

    Args:
        df (pd.DataFrame): The input DataFrame with raw scores and hand labels.

    Returns:
        tuple: (scores_df, with_hand_df)
            scores_df (pd.DataFrame): Cleaned DataFrame with NAs filled (except 'Hand_firm_rate_comb').
            with_hand_df (pd.DataFrame): Subset with only hand-labeled rows.
    """
    # list of columns and models to remove from analysis
    exclude_words = ["raw", "comb_short_1", "Titan"]
    score_cols = [
        col for col in df.columns if all(word not in col for word in exclude_words)
    ]

    scores_df = df[score_cols]
    filtered_columns = [
        col for col in scores_df.columns if any(key in col for key in model_key)
    ]

    # Convert all filtered columns to numeric, coercing errors to NaN
    for col in filtered_columns:
        scores_df[col] = pd.to_numeric(scores_df[col], errors="coerce")

    # Fill NA in all columns except 'Hand_firm_rate_comb' with column mean
    cols_to_fill = [col for col in filtered_columns if col != "Hand_firm_rate_comb"]
    scores_df[cols_to_fill] = scores_df[cols_to_fill].apply(
        lambda col: col.fillna(col.mean()), axis=0
    )

    # Subset to rows with hand labels for with_hand_df
    with_hand_df = scores_df[scores_df["Hand_firm_rate_comb"].notna()].copy()

    return scores_df, with_hand_df
