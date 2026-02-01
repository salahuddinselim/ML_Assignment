
!pip install hmmlearn
import kagglehub
import sqlite3
import pandas as pd
import numpy as np
from hmmlearn import hmm


path = kagglehub.dataset_download("hugomathien/soccer")
conn = sqlite3.connect(f"{path}/database.sqlite")


query = """
SELECT home_team_goal, away_team_goal
FROM Match
WHERE home_team_api_id = 8634 OR away_team_api_id = 8634
ORDER BY date ASC
"""
matches = pd.read_sql_query(query, conn)


def get_outcome(row):
    
    if row['home_team_goal'] > row['away_team_goal']: return 2 
    if row['home_team_goal'] == row['away_team_goal']: return 1 
    return 0 

matches['outcome'] = matches.apply(get_outcome, axis=1)
observations = matches['outcome'].values.reshape(-1, 1)


model = hmm.MultinomialHMM(n_components=3, n_iter=100, random_state=42)


model.fit(observations)


hidden_states = model.predict(observations)

print("Transition Matrix (States shifting):")
print(model.transmat_)

print("\nEmission Matrix (Results given state):")
print(model.emissionprob_)

import matplotlib.pyplot as plt

plt.figure(figsize=(15, 5))
plt.plot(hidden_states, label="Hidden Form State", color='blue', alpha=0.5)
plt.yticks([0, 1, 2], ['Slump', 'Average', 'Peak'])
plt.title("FC Barcelona Hidden Form Cycles Over Time")
plt.xlabel("Match Number")
plt.show()