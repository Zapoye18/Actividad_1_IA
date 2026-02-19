import os
import pandas as pd
from itertools import combinations

# 1: Load Dataset
downloads_folder = os.path.expanduser('~/Downloads')
data_file = os.path.join(downloads_folder, 'apriori_exercise4.xlsx')
df = pd.read_excel(data_file)

print("First 5 rows of data:")
print(df.head())
print(f"\nData shape: {df.shape}")

# Convert each row into a set of genres (ignoring empty cells)
genre_cols = ['Genre_1', 'Genre_2', 'Genre_3']
transactions = []
for _, row in df.iterrows():
    genres = set()
    for col in genre_cols:
        if pd.notna(row[col]):
            genres.add(row[col])
    transactions.append(genres)

total = len(transactions)
min_support = 0.30
print(f"\nTotal users: {total}")
print(f"Minimum support: {min_support:.0%} (at least {int(min_support * total)} users)\n")

# Part A: Find frequent itmsets: Individual genres (size-1 itemsets)
print("="*50)
print("PART 1: Frequent genres (Size-1)")
print("="*50)

all_genres = set()
for t in transactions:
    all_genres.update(t)

freq_1 = {}
for genre in sorted(all_genres):
    count = sum(1 for t in transactions if genre in t)
    support = count / total
    status = "Frequent" if support >= min_support else "Infrequent"
    print(f"    {status}: {genre:10s} -> {count}/{total} = {support:.1%}")
    if support >= min_support:
        freq_1[genre] = count
print(f"\nFrequent: {sorted(freq_1.keys())}\n")

# --- Step 2: Pairs (size-2 itemsets) ---
print("="*50)
print("STEP 2: Pairs (Size-2)")
print("="*50)

freq_2 = {}
for a,b in combinations(sorted(freq_1.keys()), 2):
    pair = frozenset([a, b])
    count = sum(1 for t in transactions if pair.issubset(t))
    support = count / total
    status = "Frequent" if support >= min_support else "Infrequent"
    print(f"    {status} {a} + {b:10s} -> {count}/{total} = {support:.1%}")
    if support >= min_support:
        freq_2[pair] = count
print(f"\nFrequent pairs: {[set(p) for p in freq_2.keys()]}\n")

# --- Step 3: Trios with pruning (size-3 itemsets) ---
print("STEP 3: Trios (with Apriori pruning)")

freq_3 = {}
for a,b,c in combinations(sorted(freq_1.keys()), 3):
    trio = frozenset([a, b, c])

    sub_pairs = list(combinations([a,b,c], 2))
    all_frequent = all(frozenset(p) in freq_2 for p in sub_pairs)

    if not all_frequent:
        print(f"{{{a}+{b}+{c}}} -> PRUNED (not all pairs frequent)")
    else:
        count = sum(1 for t in transactions if trio.issubset(t))
        support = count / total
        if support >= min_support:
            freq_3[trio] = count
            print(f"Ok {{{a}+{b}+{c}}} -> {count}/{total} = {support:.1%}")
        else:
            print(f"X {{{a}+{b}+{c}}} -> {count}/{total} = {support:.1%}")

if not freq_3:
    print("\nNo frequent trios found.")
print()

# PART B: ASSOCIATION RULES (confidence + lift)
print("STEP 4: Association Rules")

for pairs, pair_sup in freq_2.items():
    itmes = sorted(pairs)
    a,b = itmes[0], itmes[1]

    # Rule 1: a -> b
    conf_ab = pair_sup / freq_1[a]
    lift_ab = conf_ab / freq_1[b]

    # Rule 2: b -> a
    conf_ba = pair_sup / freq_1[b]
    lift_ba = conf_ba / freq_1[a]

    print(f"\n {a} -> {b}:")
    print(f"    Confidence = {conf_ab:.1%}, Lift = {lift_ab:.2f} {'Frequent genuine' if lift_ab > 1 else ' coincidence'}")

    print(f" {b} -> {a}:")
    print(f"    Confidence = {conf_ba:.1%}, Lift = {lift_ba:.2f} {'Frequent genuine' if lift_ba > 1 else ' coincidence'}")
