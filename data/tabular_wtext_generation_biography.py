import numpy as np
import pandas as pd
import os
import json
import argparse

CSV_COLS = ["age", "salary", "sex", "birth_state", "college", "degree", "occupation", "biography"]

SEX = ["male", "female"]
BIRTH_STATE = ["california", "new york", "texas", "florida", "illinois", "washington", "massachusetts", "colorado", "georgia", "arizona"]
COLLEGE = [
    "stanford university", "harvard university", "new york university",
    "university of michigan", "arizona state university", "university of central florida",
    "santa monica college", "houston community college", "ohio state university"
]
DEGREE = ["associate", "bachelor", "master", "doctoral"]
OCCUPATION = [
    "software developer", "research specialist", "healthcare practitioner", "business operations analyst",
    "education professional", "creative content professional", "technical services specialist",
    "construction professional", "customer services professional", "public services coordinator"
]

# Method B: three-peak centered salary distribution.
SALARY_CENTERS = {
    "associate": 82,
    "bachelor":  125,
    "master":    125,
    "doctoral":  178,
}
SALARY_STD = 5.0


def get_age_desc(age):
    if 21 <= age <= 30: return "in the early career stage"
    if 31 <= age <= 40: return "in the career-building stage"
    if 41 <= age <= 50: return "in the established career stage"
    if 51 <= age <= 60: return "in the advanced career stage"
    return "in the late career stage"


def get_salary_desc(sal):
    if sal <= 100: return "a comfortable income"
    if sal <= 150: return "a strong professional income"
    return "a high-level income"


def sample_salary(deg, age, occ, rng):
    """Method B: salary mode locked to the bin center, with weak age/occ
    modulation to keep some structural signal without re-introducing
    boundary cases."""
    mu = SALARY_CENTERS[deg]
    if occ in ("software developer", "healthcare practitioner"):
        mu += 4
    mu += (age - 45) * 0.3
    s = rng.normal(mu, SALARY_STD)
    return int(np.clip(s, 75, 200))


def generate_raw_data(n, rng):
    ages = rng.integers(21, 71, size=n)
    sexes = rng.choice(SEX, size=n, p=[0.5, 0.5])

    state_p = np.array([0.15, 0.12, 0.12, 0.10, 0.08, 0.08, 0.07, 0.07, 0.11, 0.10])
    states = rng.choice(BIRTH_STATE, size=n, p=state_p / state_p.sum())

    college_p = np.array([0.05, 0.05, 0.05, 0.07, 0.20, 0.15, 0.15, 0.15, 0.13])
    colleges = rng.choice(COLLEGE, size=n, p=college_p / college_p.sum())

    data = []
    for i in range(n):
        age, sex, state, coll = ages[i], sexes[i], states[i], colleges[i]

        deg_p = [0.3, 0.5, 0.15, 0.05]
        if coll in ["stanford university", "harvard university"]:
            deg_p = [0.01, 0.29, 0.4, 0.3]
        deg = rng.choice(DEGREE, p=deg_p)

        occ_p = np.ones(len(OCCUPATION))
        if deg == "doctoral":
            occ_p[OCCUPATION.index("research specialist")] += 5
            occ_p[OCCUPATION.index("education professional")] += 3
        elif deg == "associate":
            occ_p[OCCUPATION.index("customer services professional")] += 4
            occ_p[OCCUPATION.index("construction professional")] += 4
        occ = rng.choice(OCCUPATION, p=occ_p / occ_p.sum())

        salary = sample_salary(deg, age, occ, rng)

        pronoun = "He" if sex == "male" else "She"
        age_desc = get_age_desc(age)
        sal_desc = get_salary_desc(salary)

        bio = (
            f"This {sex} individual is {age_desc}. "
            f"{pronoun} earns {sal_desc}. "
            f"{pronoun} was born in {state} and completed higher education at {coll}, "
            f"earning a {deg} degree. "
            f"{pronoun} works as a {occ}."
        )

        data.append([age, salary, sex, state, coll, deg, occ, bio])

    return pd.DataFrame(data, columns=CSV_COLS)


def save_outputs(train_df, val_df, test_df, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train_df.to_csv(os.path.join(out_dir, "train_ori.csv"), index=False)
    val_df.to_csv(os.path.join(out_dir, "valid_ori.csv"), index=False)
    test_df.to_csv(os.path.join(out_dir, "test_ori.csv"), index=False)
    print(f"Datasets saved to {out_dir}")


def summarize_age_salary_bins(train_df):
    age_bins = [
        ("21-30", 21, 30, "in the early career stage"),
        ("31-40", 31, 40, "in the career-building stage"),
        ("41-50", 41, 50, "in the established career stage"),
        ("51-60", 51, 60, "in the advanced career stage"),
        ("other", None, None, "in the late career stage"),
    ]

    salary_bins = [
        ("<=100", None, 100, "a comfortable income"),
        ("101-150", 101, 150, "a strong professional income"),
        (">150", 151, None, "a high-level income"),
    ]

    n = len(train_df)

    print("\nAge interval distribution:")
    age = train_df["age"].astype(int)

    for name, lo, hi, desc in age_bins:
        if lo is None and hi is None:
            mask = ~((age >= 21) & (age <= 60))
        else:
            mask = (age >= lo) & (age <= hi)

        count = mask.sum()
        ratio = count / n * 100.0
        print(f"{name:12s} | {count:6d} | {ratio:6.2f}% | {desc}")

    print("\nSalary interval distribution:")
    sal = train_df["salary"].astype(int)

    for name, lo, hi, desc in salary_bins:
        if lo is None:
            mask = sal <= hi
        elif hi is None:
            mask = sal >= lo
        else:
            mask = (sal >= lo) & (sal <= hi)

        count = mask.sum()
        ratio = count / n * 100.0
        print(f"{name:12s} | {count:6d} | {ratio:6.2f}% | {desc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=10000, help="Total train+val samples")
    parser.add_argument("--test_size", type=int, default=2250, help="Test samples")
    parser.add_argument("--out_dir", type=str, default="data/tabular/biography")
    args = parser.parse_args()

    rng = np.random.default_rng(42)

    full_train_val = generate_raw_data(args.num_samples, rng)
    test_df = generate_raw_data(args.test_size, rng)

    train_count = int(args.num_samples * 0.9)
    train_df = full_train_val.iloc[:train_count]
    val_df = full_train_val.iloc[train_count:]

    save_outputs(train_df, val_df, test_df, args.out_dir)

    print("\n--- Summary ---")
    print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
    print("\nDegree Distribution:\n", train_df['degree'].value_counts(normalize=True))
    print("\nAverage Salary by Degree:\n", train_df.groupby('degree')['salary'].mean())

    sal = train_df['salary'].astype(int)
    print(f"Age distribution: {train_df['age'].value_counts(normalize=True)}")
    print(f"Salary distribution: {train_df['salary'].value_counts(normalize=True)}")
    print(f"Salary [min,max]=({sal.min()},{sal.max()}), mean={sal.mean():.2f}, std={sal.std():.2f}")

    summarize_age_salary_bins(train_df)
