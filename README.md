# FIFA 21 Data Analysis

This Streamlit application provides interactive data analysis and visualization for the FIFA 21 dataset. It allows users to explore various aspects of player statistics, values, and league comparisons.

## Features

- **Value Distribution Analysis**: Visualize the distribution of player values across the dataset.
- **Value by League Analysis**: Compare average player values across different leagues.
- **Interactive Interface**: Use Streamlit's interactive widgets to customize the analysis.
- **Data Cleaning**: Automatic data cleaning and outlier removal options.
- **Insightful Visualizations**: Clear and informative plots with accompanying insights.

## Key Insights

The application provides various insights into the FIFA 21 dataset, including:

### Value Distribution
- Player values are highly skewed, with a small number of players having exceptionally high values.
- The majority of players are clustered at lower values, forming the base of the distribution.
- This skewed distribution reflects the reality of the football market, where top players command significantly higher values than the average player.

### Value by League
- The English Premier League has the highest average player value, reflecting its financial power and global popularity.
- The top 5 European leagues (England, Spain, Germany, Italy, France) dominate in terms of player values.
- There's a significant drop-off in average player value after the top few leagues, highlighting financial disparities in global football.
- Leagues from smaller countries or with less global exposure tend to have lower average player values, despite potentially having high-quality players.

These insights provide a snapshot of the economic landscape in global football and the relative financial strength of different leagues.

## Requirements

- Python 3.7+
- Streamlit
- Pandas
- NumPy
- Seaborn
- Matplotlib
- SciPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/fifa21-data-analysis.git
   cd fifa21-data-analysis
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Ensure you have the FIFA 21 dataset (`data.csv`) in the project directory.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open your web browser and go to `http://localhost:8501` to view the application.

4. Use the sidebar to select different analysis options and toggle data cleaning features.

## Data

The application uses the FIFA 21 dataset, which should be named `data.csv` and placed in the project root directory. Ensure the dataset contains the following columns:

- `short_name`
- `age`
- `height_cm`
- `weight_kg`
- `club_name`
- `league_name`
- `league_rank`
- `overall`
- `potential`
- `value_eur`
- `player_positions`
- `team_position`
- `preferred_foot`
- `international_reputation`
- `release_clause_eur`
- `contract_valid_until`
- Various skill attributes (e.g., `attacking`, `defending`, `goalkeeping`, etc.)

## Contributing

Contributions to improve the analysis or add new features are welcome. Please feel free to submit a pull request or open an issue to discuss potential changes.
