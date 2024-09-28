import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. Import data from the CSV file
df = pd.read_csv('medical_examination.csv')

# 2. Add 'overweight' column by calculating BMI and determining if it's greater than 25
df['overweight'] = ((df['weight'] / ((df['height'] / 100) ** 2)) > 25).astype(int)

# 3. Normalize data by making 0 always good and 1 always bad
# For 'cholesterol' and 'gluc', set values of 1 to 0, and values greater than 1 to 1
df['cholesterol'] = (df['cholesterol'] > 1).astype(int)
df['gluc'] = (df['gluc'] > 1).astype(int)

# 4
def draw_cat_plot():
    # 5. Create DataFrame for cat plot using 'pd.melt'
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']
    )

    # 6. Group and reformat the data to show counts of each feature
    df_cat = df_cat.groupby(['cardio', 'variable', 'value']).size().reset_index(name='total')

    # 7. Draw the catplot with 'sns.catplot()'
    g = sns.catplot(
        x='variable',
        y='total',
        hue='value',
        col='cardio',
        data=df_cat,
        kind='bar'
    )

    # 8. Get the figure from the plot for saving
    fig = g.fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11. Clean the data by filtering out incorrect patient segments
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &  # Diastolic pressure should be less than or equal to systolic
        (df['height'] >= df['height'].quantile(0.025)) &  # Height above 2.5th percentile
        (df['height'] <= df['height'].quantile(0.975)) &  # Height below 97.5th percentile
        (df['weight'] >= df['weight'].quantile(0.025)) &  # Weight above 2.5th percentile
        (df['weight'] <= df['weight'].quantile(0.975))    # Weight below 97.5th percentile
    ]

    # 12. Calculate the correlation matrix
    corr = df_heat.corr()

    # 13. Generate a mask for the upper triangle to avoid redundancy
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14. Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(12, 12))

    # 15. Draw the heatmap using seaborn
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt='.1f',
        center=0,
        square=True,
        cbar_kws={'shrink': 0.5}
    )

    # 16
    fig.savefig('heatmap.png')
    return fig
