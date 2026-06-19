"""
generate_dataset.py
--------------------
Generates a realistic synthetic placement dataset of 1000+ student records.

FEATURE EXPLANATIONS
---------------------
CGPA                    : Cumulative GPA on a 10-point scale (academic performance).
Tenth_Percentage        : Marks scored in 10th grade board exam (0-100).
Twelfth_Percentage      : Marks scored in 12th grade board exam (0-100).
Internships             : Number of internships completed (0-4).
Projects                : Number of academic/personal projects completed (0-10).
Coding_Skills           : Self/test-assessed coding ability score (0-100), e.g. from
                           platforms like LeetCode/HackerRank assessments.
Communication_Skills    : Score from mock interviews / soft-skill assessments (0-100).
Aptitude_Score          : Score in quantitative/logical aptitude tests (0-100).
Technical_Skills        : Score reflecting depth of technical knowledge (0-100).
Backlogs                : Number of pending/failed subjects (0-5). Strongly hurts placement.
Certifications          : Number of relevant certifications completed (0-8).
Hackathon_Participation : Number of hackathons participated in (0-5).
Placement_Status        : TARGET. 1 = Placed, 0 = Not Placed.

The target is generated using a weighted logistic function of the features plus
random noise, so relationships are realistic but not perfectly deterministic
(mirrors real-world placement outcomes).
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200  # >1000 records


def generate_dataset(n=N):
    cgpa = np.clip(np.random.normal(7.2, 1.0, n), 4.0, 10.0)
    tenth = np.clip(np.random.normal(78, 10, n), 40, 100)
    twelfth = np.clip(np.random.normal(75, 10, n), 40, 100)
    internships = np.random.poisson(1.1, n).clip(0, 4)
    projects = np.random.poisson(2.5, n).clip(0, 10)
    coding = np.clip(np.random.normal(60, 18, n), 0, 100)
    communication = np.clip(np.random.normal(65, 15, n), 0, 100)
    aptitude = np.clip(np.random.normal(60, 17, n), 0, 100)
    technical = np.clip(np.random.normal(60, 18, n), 0, 100)
    backlogs = np.random.choice([0, 1, 2, 3, 4, 5], n, p=[0.55, 0.2, 0.12, 0.07, 0.04, 0.02])
    certifications = np.random.poisson(1.5, n).clip(0, 8)
    hackathons = np.random.poisson(0.8, n).clip(0, 5)

    # Weighted score driving placement probability (realistic signal)
    z = (
        0.55 * (cgpa - 7)
        + 0.015 * (tenth - 75)
        + 0.015 * (twelfth - 75)
        + 0.35 * internships
        + 0.18 * projects
        + 0.025 * (coding - 60)
        + 0.02 * (communication - 60)
        + 0.02 * (aptitude - 60)
        + 0.02 * (technical - 60)
        - 0.6 * backlogs
        + 0.15 * certifications
        + 0.12 * hackathons
        + np.random.normal(0, 1.0, n)  # noise
    )
    prob = 1 / (1 + np.exp(-z))
    placement = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "CGPA": cgpa.round(2),
        "Tenth_Percentage": tenth.round(2),
        "Twelfth_Percentage": twelfth.round(2),
        "Internships": internships,
        "Projects": projects,
        "Coding_Skills": coding.round(1),
        "Communication_Skills": communication.round(1),
        "Aptitude_Score": aptitude.round(1),
        "Technical_Skills": technical.round(1),
        "Backlogs": backlogs,
        "Certifications": certifications,
        "Hackathon_Participation": hackathons,
        "Placement_Status": placement,
    })

    # Inject a small % of missing values to make preprocessing realistic
    for col in ["CGPA", "Coding_Skills", "Communication_Skills", "Aptitude_Score"]:
        idx = df.sample(frac=0.02, random_state=1).index
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("placement_dataset.csv", index=False)
    print(f"Generated {len(df)} records.")
    print(df["Placement_Status"].value_counts(normalize=True))
