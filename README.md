# user-adoption-analysis

**Business Problem:**
E-learning platforms invest heavily in student acquisition — but 31.2% of students withdraw before completing their course. This project investigates the behavioral and demographic patterns that predict student withdrawal, and translates those findings into actionable recommendations for the platform.

**Key Question**: What drives students to adopt the platform long-term vs. abandon it — and when can we intervene?

**Dataset**:
Open University Learning Analytics Dataset (OULAD)

32,593 students across 7 course modules
Real UK university e-learning platform data
Full semester of behavioral and assessment data
Source: Kaggle — OULAD Dataset

**Key Findings**:
    Finding 1 — Early Non-Submission Predicts Churn
    
    5,847 students never submitted a single assessment
    79.5% of them withdrew vs 31.2% overall churn rate
    2.5x higher churn risk for non-submitters
    SOLUTION: Flag non-submitters by Day 7 for tutor outreach
    
    Finding 2 — Massive Gap Between Modules
    
    Best module (GGG): 11.5% churn
    Worst module (CCC): 44.5% churn
    33 percentage point gap on the same platform
    SOLUTION: Urgent audit of CCC course design
    
    Finding 3 — The Week 7 Cliff
    
    Retention drops from 70.9% → 31.6% in a single week
    Caused by multiple high-weight assessment deadlines (18-20% weight TMAs)
    Students who aren't prepared withdraw rather than fail
    SOLUTION: Maximum support push in weeks 5-6 before cliff hits
    
    Finding 4 — Behavior Beats Demographics
    
    Top churn predictors: avg_submission_day, num_assessments, avg_score
    Demographics (age, gender, region) rank lowest in importance
    Platform cannot predict churn at signup — but CAN within first 2-3 weeks
    SOLUTION: Early warning system based on behavior, not demographics
    
    Finding 5 — Repeat Attempters at Higher Risk
    
    First attempt: 30.6% churn
    Third attempt: 36.6% churn
    Each additional attempt adds ~2% churn risk
    SOLUTION: Mandatory counseling for students with 2+ previous attempts



