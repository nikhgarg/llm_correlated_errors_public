firm_rate_p0 = f"""

Human: You are going to assess a resume's suitability to a job description for three metrics:

Overall fit, category fit, and skill fit. 

You will provide me a score for each category from 1 to 5 where 1 signals lowest fit and 5 signals highest fit.

The output format is a json. For instance: 

{{
"Overall Fit": 4,
"Category Fit": 3,
"Skill Fit": 5
}}

"""

firm_rate_p1 = f"""

Human: I am going to define three different metrics for you. 

These metrics will be used to assess the suitability of someone's resume to a company's job description. 
1. Overall fit: This metric evaluates the comprehensive alignment of a resume with a job description,
accounting for the candidate's background, experiences, and qualifications. 

2. Category fit: This metric assesses the relevance of the candidate's primary field or industry background 
to the job's target audience or sector. For example, evaluating a resume from an arts professional with Python 
skills for a Python developer role would likely result in a lower score due to mismatched primary disciplines.

3. Skill fit:  This metric measures the match between the specific skills and expertise listed on the resume 
and those required by the job. It considers both the level and the relevance of the skills, providing a score 
based on how well the candidate's abilities meet the job's demands in terms of proficiency and experience. 

You will provide me a score for each category from 1 to 5 where 1 signals lowest fit and 5 signals highest fit.

The output format is a json. For instance: 

{{
"Overall Fit": 4,
"Category Fit": 3,
"Skill Fit": 5
}}

"""

firm_rate_p2 = f"""

Human: Below, you will find specific metrics to evaluate the alignment of a resume with a job description. Each metric should be scored from 1 to 5, where 1 indicates a low alignment and 5 indicates a high alignment.

1. Overall Fit: Evaluate how comprehensively the resume aligns with the job description. Consider the candidate's background, experiences, and qualifications. For instance, a score of 5 means the resume matches perfectly with all aspects of the job description.

2. Category Fit: Assess the relevance of the candidate's primary field or industry relative to the job's sector. For example, a resume from an arts professional with incidental Python experience applying for a Python developer role might score low, indicating a mismatch in primary discipline.

3. Skill Fit: Measure how well the candidate's listed skills and expertise match the job requirements. Consider both the relevance and the proficiency level of these skills. A high score indicates that the candidate possesses all the necessary skills at the required proficiency levels.

Please provide a score for each category based on the following job description and resume. Additionally, if certain aspects are unclear, specify what additional information would be helpful.

**Output Format**: JSON, as shown:
{{
    "Overall Fit": 4,
    "Category Fit": 3,
    "Skill Fit": 5
}}

"""

firm_rate_lenient = f"""

Human: I am going to define three different metrics for you. 

These metrics will be used to assess the suitability of someone's resume to a company's job description. 
1. Overall fit: This metric evaluates the comprehensive alignment of a resume with a job description,
accounting for the candidate's background, experiences, and qualifications. While evaluating, consider 
the potential for growth and the candidate's overall potential to adapt and excel in the role.

2. Category fit: This metric assesses the relevance of the candidate's primary field or industry background 
to the job's target audience or sector. For example, evaluating a resume from an arts professional with Python 
skills for a Python developer role would likely result in a lower score due to mismatched primary disciplines.
However, consider the transferable skills and the potential for the candidate to excel in a new discipline rather 
than focusing solely on mismatched primary disciplines.

3. Skill fit:  This metric measures the match between the specific skills and expertise listed on the resume 
and those required by the job. It considers both the level and the relevance of the skills, providing a score 
based on how well the candidate's abilities meet the job's demands in terms of proficiency and experience. 
Also, consider the candidate's ability to learn and grow in the required skills.

You will provide me a score for each category from 1 to 5 where 1 signals lowest fit and 5 signals 
highest fit. Be lenient in your assessment and give higher scores where potential and adaptability 
are evident.

Your output format is a json file and must follow this example:

{{
"Overall Fit": 4,
"Category Fit": 3,
"Skill Fit": 5
}}

"""

firm_rate_comb_short = f"""

Human: You are going to evaluate the fit of a resume to a job description using a single score on a scale from 1 to 10. When grading, you will consider overall fit, category fit, and skill fit. Your output format is a json file and must follow this example:

{{
"Score": 7
}}

"""

firm_rate_comb_short_1 = f"""

Human: You are a hiring manager who must determine which job applicants to interview. Given a job description and a resume, rate the resume on a scale from 1 to 10. 

Your output format is a json file and must follow this example:

{{
"Score": 7
}}
"""

firm_rate_comb = f"""

Human: You are going to evaluate the fit of a resume to a job description using a single score on a scale from 1 to 10. When grading, consider the following:

1. Overall Fit: How well does the candidate's entire profile align with job's requirements?
2. Category Fit: Is the candidate's primary field or industry background relevant to the job's target sector or audience? How closely does their previous experience align with the industry the job is in?
3. How well do the candidate's skills and expertise match the specific skills required by the job? Consider both the relevance and proficiency of these skills in relation to the job's demands.

This is the grading scale: 

1 - Extremely Poor Fit: The resume shows virtually no alignment with the job description across all dimensions. The candidate's overall background, industry experience, and skill set are largely irrelevant or mismatched for the role.

3 - Poor Fit: The candidate demonstrates minimal relevance to the job description. While there might be a few aligned skills or some related experience, the overall, category, and skill fit are significantly lacking, making the candidate an unlikely match.

5 - Average Fit: The resume meets some basic requirements of the job description. The candidate shows moderate alignment in terms of overall background, industry relevance, and necessary skills, but there are noticeable gaps or weaknesses in several key areas.

7 - Good Fit: The candidate's resume aligns well with the job description across most dimensions. Their background, industry experience, and skill set are relevant, and they possess the necessary qualifications to perform the role effectively, though there may be minor gaps.

10 - Excellent Fit: The resume shows perfect alignment with the job description in every aspect. The candidate's background, industry experience, and skills are not only relevant but also demonstrate a high level of proficiency and expertise, making them an ideal match for the role.

Your output format is a json file and must follow this example:

{{
"Score": 7
}}


"""

firm_rate_comb1 = f"""

Human: You are a recruiter evaluating a resume for a job description, and judge if this resume will be good at the job using a single score on a scale from 1 to 10. When grading, consider the following:

1. Overall fit: This metric evaluates the comprehensive alignment of a resume with a job description,
accounting for the candidate's background, experiences, and qualifications. While evaluating, consider 
the potential for growth and the candidate's overall potential to adapt and excel in the role.

2. Category fit: This metric assesses the relevance of the candidate's primary field or industry background 
to the job's target audience or sector. For example, evaluating a resume from an arts professional with Python 
skills for a Python developer role would likely result in a lower score due to mismatched primary disciplines.
However, consider the transferable skills and the potential for the candidate to excel in a new discipline rather 
than focusing solely on mismatched primary disciplines.

3. Skill fit:  This metric measures the match between the specific skills and expertise listed on the resume 
and those required by the job. It considers both the level and the relevance of the skills, providing a score 
based on how well the candidate's abilities meet the job's demands in terms of proficiency and experience. 
Also, consider the candidate's ability to learn and grow in the required skills.

This is the grading scale: 

10 - Excellent Fit: The resume shows perfect alignment with the job description in every aspect. The candidate's background, industry experience, and skills are not only relevant but also demonstrate a high level of proficiency and expertise, making them an ideal match for the role.

7 - Good Fit: The candidate's resume aligns well with the job description across most dimensions. Their background, industry experience, and skill set are relevant, and they possess the necessary qualifications to perform the role effectively, though there may be minor gaps.

5 - Average Fit: The resume meets some basic requirements of the job description. The candidate shows moderate alignment in terms of overall background, industry relevance, and necessary skills, but there are noticeable gaps or weaknesses in several key areas.

3 - Poor Fit: The candidate demonstrates minimal relevance to the job description. While there might be a few aligned skills or some related experience, the overall, category, and skill fit are significantly lacking, making the candidate an unlikely match.

1 - Extremely Poor Fit: The resume shows virtually no alignment with the job description across all dimensions. The candidate's overall background, industry experience, and skill set are largely irrelevant or mismatched for the role.

Your output format is a json file and must follow this example:

{{
"Score": 
}}

"""

firm_rate_comb2 = f"""

Human: You are going to evaluate the fit of a resume to a job description using a single score on a scale from 1 to 10. When grading, consider the following:

1. Overall fit: This metric evaluates how well the candidate's resume aligns with the job description, taking into account their background, experiences, and qualifications. Consider not only their current suitability but also their potential for growth, adaptability, and ability to thrive in the role.

2. Category fit: This metric assesses the relevance of the candidate's primary field or industry experience to the job's sector. For example, if a resume comes from a professional in a different field but with some relevant transferable skills, consider their potential to succeed in the new field, rather than focusing solely on mismatched backgrounds.

3. Skill fit: This metric evaluates the match between the candidate's listed skills and those required by the job. Consider both the relevance and proficiency of these skills, but also factor in the candidate's ability to learn and grow in areas where there may be gaps, ensuring the evaluation takes into account future potential.

This is the grading scale:

10 - Excellent Fit: The resume aligns very closely with the job description in all key areas. The candidate's background, industry experience, and skills are not only relevant but also demonstrate high proficiency, making them a strong match for the role.

7 - Good Fit: The resume shows strong alignment with the job description in most areas. The candidate's background and skills are largely relevant, and while there may be a few gaps, they have the necessary qualifications and potential to perform the role effectively.

5 - Average Fit: The resume meets some of the key requirements of the job description. While the candidate may not have perfect alignment, there is still moderate relevance in terms of background, skills, and experience, though some key areas may be lacking.

3 - Poor Fit: The candidate shows limited relevance to the job description. Although there may be a few transferable skills or some related experience, significant gaps exist across background, skills, or industry relevance, making the candidate a less likely match.

1 - Extremely Poor Fit: The resume shows little to no alignment with the job description. The candidate's background, experience, and skills are largely unrelated to the job's requirements, making them an unlikely match for the role.

Your output format is a json file and must follow this example: 

Your output format is a json file and must follow this example:

{{
"Score": 7
}}
"""

app_rate_comb_short = f"""

Human:  You are a job seeker with the following resume. Rate the following job description for how much you would like to work there using a single score on a scale from 1 to 10. Your output format is a json file and must follow this example:

Your output format is a json file and must follow this example:

{{
"Score": 7
}}
"""

app_rate_comb = f"""

Human: You are a job seeker with the following resume. Rate the following job description for how much you would like to work there using a single score on a scale from 1 to 10. When grading, consider the following:

1. Overall Fit: How well does the job's requirements align with the candidate's entire profile, including their experience, skills, and career trajectory?
2. Category Fit: Does the job's target sector or industry match the candidate's primary field or industry background? How closely do the job's requirements align with the candidate's previous roles in the same or related industries?
3. Skill Fit: How well do the job's specific skill requirements match the candidate's demonstrated skills and expertise? Consider both the relevance and proficiency of these skills as they pertain to the candidate's previous roles and achievements.

This is the grading scale:

1 - Extremely Poor Fit: The job description shows virtually no alignment with the candidate's resume across all dimensions. The job's industry, requirements, and skills are largely irrelevant or mismatched for the candidate's background.

3 - Poor Fit: The job description demonstrates minimal relevance to the candidate's resume. While there might be a few aligned requirements or some related industry overlap, the overall, industry, and skill fit are significantly lacking, making the job an unlikely match for the candidate.

5 - Average Fit: The job description meets some basic requirements of the candidate's resume. The job shows moderate alignment in terms of industry relevance and necessary skills, but there are noticeable gaps or weaknesses in several key areas that might make the candidate underqualified.

7 - Good Fit: The job description aligns well with the candidate's resume across most dimensions. The job's requirements, industry relevance, and necessary skills are a strong match, and the candidate is likely well-suited to perform the role effectively, though there may be minor gaps.

10 - Excellent Fit: The job description shows perfect alignment with the candidate's resume in every aspect. The job's industry, requirements, and skills are not only relevant but also perfectly match the candidate's background and expertise, making it an ideal role for them.

Your output format is a json file and must follow this example:

{{
"Score": 7
}}

"""

app_rate_p0 = f"""

Human: I am going to define three different metrics for you. 

You are a job applicant with a resume, and you will assess the suitability of a company's job description 
to your resume using the three metrics.

1. Overall fit: This metric evaluates the comprehensive alignment of a job description with your resume, 
accounting for your background, experiences, and qualifications.

2. Category fit: This metric assesses the relevance of the job's target audience or sector to your primary 
field or industry background. For example, evaluating a job description for a Python developer role with your 
background as an arts professional with Python skills would likely result in a lower score due to mismatched 
primary disciplines.

3. Skill fit: This metric measures the match between the specific skills and expertise required by the job 
and those listed on your resume. It considers both the level and the relevance of the skills, providing a 
score based on how well the job's demands meet your abilities in terms of proficiency and experience.

You will provide a score for each category from 1 to 5, where 1 signals lowest fit and 5 signals highest fit.

Your output format is a json file and must follow this example:

{{
"Overall Fit": 4,
"Category Fit": 3,
"Skill Fit": 5
}}

"""

firm_compare_p0 = f"""

Human: You are going to read two resumes and a job description. You must choose a number 1 or 2, 
between the first and second resume to see which resume is a better fit for the job description 
and provide a brief explanation.

Your output format is a JSON file and must follow these examples: 

{{
    "Resume": "1",
    "Explanation": "Resume 1 is a better fit for the job description because it has extensive experience in project management and team leadership. Resume 2, although strong in technical skills, does not demonstrate the same level of leadership experience required for the role."
}},

{{
    "Resume": "2",
    "Explanation": "Resume 2 is a better fit for the job description as it showcases a robust background in digital marketing and SEO strategies, which are critical for the position. Resume 1 has some relevant experience but lacks the depth in marketing expertise that Resume 2 provides."
}},

{{
    "Resume": "1",
    "Explanation": "Resume 1 is a better fit for the job description because it highlights significant experience in financial analysis and risk management. Resume 2, while competent, does not have the same level of financial expertise needed for this role."
}},

{{
    "Resume": "2",
    "Explanation": "Resume 2 is a better fit for the job description due to its strong background in healthcare administration and patient management systems. Resume 1 lacks comparable experience in the healthcare sector, making Resume 2 the preferred choice."
}},

{{
    "Resume": "1",
    "Explanation": "Resume 1 is a better fit for the job description because it includes extensive experience in educational program development and teaching. Resume 2, although having some relevant skills, does not demonstrate the same level of experience in the education field."
}}

"""

app_compare_p0 = f"""

Human: You are a job applicant with a resume and have the option to choose between
two job descriptions. You must choose a number 1 or 2, between the first and second job descriptions
to see which job is a better fit for your resume and provide a brief explanation.

Your output format is a json file and must follow these examples: 

{{
    "Job": "1",
    "Explanation": "Job 1 is a better fit because it offers more opportunities for career growth in the field of data science. Job 2, while interesting, focuses more on traditional IT roles and lacks the same level of data-centric projects."
}},

{{
    "Job": "2",
    "Explanation": "Job 2 is a better fit due to its strong emphasis on innovative software development and its collaborative work environment. Job 1, though appealing, does not provide the same level of innovation and teamwork opportunities."
}},

{{
    "Job": "1",
    "Explanation": "Job 1 is a better fit because it aligns more closely with experience in cloud computing and distributed systems. Job 2, although offering a good salary, is more focused on on-premises solutions, which do not align with the applicant's expertise."
}},

{{
    "Job": "2",
    "Explanation": "Job 2 is a better fit due to its comprehensive benefits package and work-life balance policies. Job 1, while offering a competitive salary, does not match the same level of benefits and flexible working hours as Job 2."
}},

{{
    "Job": "1",
    "Explanation": "Job 1 is a better fit because it is located in a city with a lower cost of living and offers remote work options. Job 2, despite being in a high-profile company, requires relocation to a city with a higher cost of living, which may not be desirable."
}}

"""