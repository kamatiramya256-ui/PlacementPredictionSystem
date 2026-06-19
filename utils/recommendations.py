"""
recommendations.py
-------------------
Rule-based Skill Gap Analysis Engine.

Each rule checks a feature against a threshold and, if the student falls
below it, appends a targeted recommendation. Thresholds are based on
reasonable placement-readiness benchmarks and can be tuned.
"""

RULES = [
    {
        "feature": "Coding_Skills", "threshold": 60, "below": True,
        "area": "Coding / DSA",
        "message": "Your coding score is below par. Practice Data Structures & "
                    "Algorithms daily on platforms like LeetCode or GeeksforGeeks, "
                    "aiming for at least 150 solved problems before placements.",
    },
    {
        "feature": "Communication_Skills", "threshold": 60, "below": True,
        "area": "Communication",
        "message": "Your communication score needs work. Join group discussions, "
                    "practice mock interviews, and consider a communication/soft-"
                    "skills training course.",
    },
    {
        "feature": "Aptitude_Score", "threshold": 55, "below": True,
        "area": "Aptitude",
        "message": "Your aptitude score is low. Dedicate time to quantitative "
                    "reasoning, logical reasoning, and verbal ability practice "
                    "(e.g. via IndiaBix or RS Aggarwal).",
    },
    {
        "feature": "Projects", "threshold": 2, "below": True,
        "area": "Portfolio",
        "message": "You have very few projects. Build 2-3 solid portfolio "
                    "projects that demonstrate real-world problem solving and "
                    "host them on GitHub with good READMEs.",
    },
    {
        "feature": "Internships", "threshold": 1, "below": True,
        "area": "Industry Experience",
        "message": "You lack internship experience. Apply for internships via "
                    "LinkedIn, Internshala, or campus placement cells to gain "
                    "practical exposure.",
    },
    {
        "feature": "Technical_Skills", "threshold": 60, "below": True,
        "area": "Technical Depth",
        "message": "Strengthen your core technical fundamentals (CS basics: "
                    "OS, DBMS, Networks, OOP) since these are heavily tested in "
                    "interviews.",
    },
    {
        "feature": "Certifications", "threshold": 1, "below": True,
        "area": "Certifications",
        "message": "Consider earning relevant certifications (e.g. AWS, Google "
                    "Cloud, Coursera/NPTEL courses) to strengthen your resume.",
    },
    {
        "feature": "Backlogs", "threshold": 0, "below": False,  # flag if ABOVE
        "area": "Academics",
        "message": "Clear your pending backlogs as a priority — many recruiters "
                    "filter out candidates with active backlogs.",
    },
    {
        "feature": "CGPA", "threshold": 6.5, "below": True,
        "area": "Academics",
        "message": "Your CGPA is below typical recruiter cutoffs (often 6.5-7.0). "
                    "Focus on improving grades in upcoming semesters.",
    },
    {
        "feature": "Hackathon_Participation", "threshold": 1, "below": True,
        "area": "Practical Exposure",
        "message": "Participate in hackathons to build practical, collaborative "
                    "project experience and stand out to recruiters.",
    },
]


def analyze_skill_gap(student: dict):
    """
    student: dict with keys matching dataset feature names.
    Returns a list of {area, message} recommendation dicts for every
    weak area detected, plus a summary of strengths.
    """
    weaknesses = []
    strengths = []

    for rule in RULES:
        val = student.get(rule["feature"])
        if val is None:
            continue
        is_weak = (val < rule["threshold"]) if rule["below"] else (val > rule["threshold"])
        if is_weak:
            weaknesses.append({"area": rule["area"], "message": rule["message"]})
        else:
            strengths.append(rule["area"])

    if not weaknesses:
        weaknesses.append({
            "area": "General",
            "message": "No major weaknesses detected — maintain consistency and "
                       "start interview preparation early.",
        })

    return {"weaknesses": weaknesses, "strengths": sorted(set(strengths))}
