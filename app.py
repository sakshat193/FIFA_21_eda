import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

plt.style.use('dark_background')

# Set page configuration
st.set_page_config(page_title="FIFA 21 Data Analysis", layout="wide")

st.markdown(
    """
    <style>
    /* Primary background color */
    .css-1d391kg { 
        background-color: #000000;
    }

    /* Secondary background color */
    .css-1avcm0n, .css-1siy2j7 { 
        background-color: #171717;
    }

    /* Text color */
    .css-145kmo2, .css-10trblm {
        color: #ffffff;
    }

    /* Sidebar background color */
    .css-1d391kg .css-1r6slb0 {
        background-color: #171717;
    }

    /* Widget background color */
    .css-1avcm0n .css-1e5imcs, .css-1siy2j7 .css-1e5imcs {
        background-color: #171717;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data.csv')
    return df

df = load_data()

# Data Cleaning
def clean_data(df):
    threshold = len(df) * 0.30
    df = df.dropna(thresh=len(df) - threshold, axis=1)
    df = df.drop_duplicates(subset=['short_name'])
    
    skill_categories = {
        'goalkeeping': ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking',
                        'goalkeeping_positioning', 'goalkeeping_reflexes'],
        'defending': ['defending_standing_tackle', 'defending_sliding_tackle'],
        'mentality': ['mentality_aggression', 'mentality_interceptions', 'mentality_positioning',
                      'mentality_vision', 'mentality_penalties', 'mentality_composure'],
        'power': ['power_shot_power', 'power_jumping', 'power_stamina',
                  'power_strength', 'power_long_shots'],
        'movement': ['movement_acceleration', 'movement_sprint_speed', 'movement_agility',
                     'movement_reactions', 'movement_balance'],
        'skill': ['skill_dribbling', 'skill_curve', 'skill_fk_accuracy',
                  'skill_long_passing', 'skill_ball_control'],
        'attacking': ['attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 
                      'attacking_volleys', 'attacking_short_passing']
    }
    
    for category, cols in skill_categories.items():
        if all(col in df.columns for col in cols):
            df[category] = df[cols].mean(axis=1).round(2)
            df = df.drop(cols, axis=1)
    
    df['bmi'] = (df['weight_kg'] / ((df['height_cm'] / 100) ** 2)).round(2)
    df = df.drop(columns=['height_cm', 'weight_kg'])
    
    columns_to_keep = [
        'short_name', 'age', 'bmi', 'club_name', 'league_name', 'league_rank',
        'overall', 'potential', 'value_eur', 'player_positions', 'team_position', 'preferred_foot', 
        'international_reputation', 'release_clause_eur', 'contract_valid_until', 
        'goalkeeping', 'defending', 'movement', 'attacking'
    ]
    df = df.loc[:, columns_to_keep]
    
    return df

def remove_outliers(df, columns):
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    return df

# Sidebar for options
st.sidebar.title("FIFA 21 Data Analysis")

# Toggle for removing outliers
remove_outliers_option = st.sidebar.checkbox("Remove Outliers", value=True)

# Perform data cleaning
df_cleaned = clean_data(df)

if remove_outliers_option:
    df_cleaned = remove_outliers(df_cleaned, ['value_eur'])

# Sidebar for navigation
analysis_option = st.sidebar.selectbox("Choose Analysis Type", 
                                       ("Age Distribution", "Age vs Player Positions", 
                                        "Value EUR vs Player Positions", "Value EUR vs Release Clause",
                                        "Value EUR vs Top 7 Performances", "Value EUR vs Skills", 
                                        "Correlation Analysis", "Value Distribution", "Value by League"))

# Age Distribution Analysis
if analysis_option == "Age Distribution":
    st.title("Age Distribution of Players")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_cleaned['age'], bins=range(15, 60, 5), kde=True, ax=ax)
    ax.set_title('Age Distribution of Players')
    ax.set_xlabel('Age')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Age vs Player Positions Analysis
elif analysis_option == "Age vs Player Positions":
    st.title("Age vs Player Positions")
    position_mapping = {
        'RWB': 'RB', 'LWB': 'LB',
        'RM': 'CM', 'LM': 'CM'
    }
    df_cleaned['primary_position'] = df_cleaned['player_positions'].apply(lambda x: position_mapping.get(x.split(',')[0].strip(), x.split(',')[0].strip()))
    position_age = df_cleaned.groupby('primary_position')['age'].mean().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='age', y='primary_position', data=position_age, ax=ax)
    ax.set_title('Average Age by Primary Position')
    ax.set_xlabel('Average Age')
    ax.set_ylabel('Player Position')
    st.pyplot(fig)

# Value_EUR vs Player Positions Analysis
elif analysis_option == "Value EUR vs Player Positions":
    st.title("Value EUR vs Player Positions")
    position_value = df_cleaned.groupby('primary_position')['value_eur'].mean().sort_values(ascending=False).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='value_eur', y='primary_position', data=position_value, ax=ax)
    ax.set_title('Average Player Value by Primary Position')
    ax.set_xlabel('Average Value (EUR)')
    ax.set_ylabel('Player Position')
    st.pyplot(fig)

# Value_EUR vs Release Clause Analysis
elif analysis_option == "Value EUR vs Release Clause":
    st.title("Value EUR vs Release Clause")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x='value_eur', y='release_clause_eur', data=df_cleaned, ax=ax)
    ax.set_title("Player Value (EUR) vs Release Clause (EUR)")
    ax.set_xlabel("Player Value (EUR)")
    ax.set_ylabel("Release Clause (EUR)")
    st.pyplot(fig)

# Value_EUR vs Top 7 Performances Analysis
elif analysis_option == "Value EUR vs Top 7 Performances":
    st.title("Value EUR vs Top 7 Performances")
    top_players = df_cleaned.nsmallest(7, 'league_rank')
    skills = ['goalkeeping', 'defending', 'movement', 'attacking']

    fig, ax1 = plt.subplots(figsize=(15, 6))
    bar_width = 0.2
    index = np.arange(len(top_players))

    for i, skill in enumerate(skills):
        ax1.bar(index + i*bar_width, top_players[skill], bar_width, label=skill)

    ax2 = ax1.twinx()
    ax2.plot(index + 1.5*bar_width, top_players['value_eur'], 'r-', marker='o', label='Value EUR')

    ax1.set_xlabel('Players')
    ax1.set_ylabel('Skill Value')
    ax2.set_ylabel('Value EUR', color='r')
    ax1.set_xticks(index + 1.5*bar_width)
    ax1.set_xticklabels(top_players['short_name'], rotation=45, ha='right')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

    plt.title('Top 7 Players by League Rank, Skills, and Value')
    plt.tight_layout()
    st.pyplot(fig)

# Value_EUR vs Skills Analysis
elif analysis_option == "Value EUR vs Skills":
    st.title("Value EUR vs Skills")
    skills = ['goalkeeping', 'defending', 'movement', 'attacking']
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, skill in enumerate(skills):
        sns.scatterplot(x=skill, y='value_eur', data=df_cleaned, ax=axes[i])
        sns.regplot(x=skill, y='value_eur', data=df_cleaned, ax=axes[i], scatter=False, color='r')
        axes[i].set_title(f'Player Value (EUR) vs {skill.capitalize()}')
        axes[i].set_xlabel(skill.capitalize())
        axes[i].set_ylabel('Player Value (EUR)')
        
        corr, _ = stats.pearsonr(df_cleaned[skill], df_cleaned['value_eur'])
        axes[i].annotate(f'Correlation: {corr:.2f}', xy=(0.05, 0.95), xycoords='axes fraction')

    plt.tight_layout()
    st.pyplot(fig)

# Correlation Analysis
elif analysis_option == "Correlation Analysis":
    st.title("Correlation Analysis")
    corr_columns = ['age', 'bmi', 'overall', 'potential', 'international_reputation', 
                    'goalkeeping', 'defending', 'movement', 'attacking']
    corr_matrix = df_cleaned[corr_columns + ['value_eur']].corr()['value_eur'].sort_values(ascending=False)
    st.write("Correlations with Player Value:")
    st.write(corr_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df_cleaned[corr_columns + ['value_eur']].corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax)
    ax.set_title('Correlation Heatmap: Player Attributes vs Value')
    st.pyplot(fig)

# Value Distribution
elif analysis_option == "Value Distribution":
    st.title("Value Distribution")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df_cleaned['value_eur'], kde=True, bins=50, ax=ax)
    ax.set_title('Distribution of Player Values')
    ax.set_xlabel('Player Value (EUR)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

# Value by League Analysis
elif analysis_option == "Value by League":
    st.title("Value by League")
    league_values = df_cleaned.groupby('league_name')['value_eur'].agg(['mean', 'median', 'count']).sort_values('mean', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=league_values.index, y='mean', data=league_values, ax=ax)
    ax.set_title('Top 10 Leagues by Average Player Value')
    ax.set_xlabel('League')
    ax.set_ylabel('Average Player Value (EUR)')
    ax.set_xticklabels(league_values.index, rotation=45, ha='right')
    st.pyplot(fig)
