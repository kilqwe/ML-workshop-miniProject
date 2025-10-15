import pandas as pd

df = pd.read_csv('PL_MP\Fifa 23 Players Data.csv')

columns = [
    "Full Name",
    "Overall",
    "Age",
    "Best Position",
    "Club Name",
    "Skill Moves",
    "Pace Total",
    "Shooting Total",
    "Passing Total",
    "Dribbling Total",
    "Defending Total",
    "Physicality Total",
    "Heading Accuracy",
    "Jumping",
    "LongPassing",
    "Goalkeeper Diving",
    "Goalkeeper Handling",
    " Goalkeeper Kicking",
    "Goalkeeper Positioning",
    "Goalkeeper Reflexes",
]

df = df[columns]
df.to_csv('PL_MP\cleaned_fifa23.csv', index=False)

fd = pd.read_csv('PL_MP\cleaned_fifa23.csv')
print(fd.head())