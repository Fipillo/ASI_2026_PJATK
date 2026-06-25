# Code export from notebooks/nba_modeling.ipynb
# Reference file for code review and refactoring

# %% Cell 1
import os

try:
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_DIR = '/content/drive/MyDrive/Colab Notebooks'
except ImportError:
    DATA_DIR = os.getcwd()


# %% Cell 2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from autogluon.tabular import TabularDataset, TabularPredictor

# %% Cell 4
games = pd.read_csv(f'{DATA_DIR}/games.csv')
details = pd.read_csv(f'{DATA_DIR}/games_details.csv', low_memory=False)

games['GAME_DATE_EST'] = pd.to_datetime(games['GAME_DATE_EST'])
games = games.sort_values('GAME_DATE_EST')

team_stats = details.groupby(['GAME_ID', 'TEAM_ID']).agg({
    'PTS': 'sum',
    'AST': 'sum',
    'REB': 'sum',
    'FG_PCT': 'mean',
    'FT_PCT': 'mean',
    'FG3_PCT': 'mean'
}).reset_index()

print(f"Games loaded: {len(games)} rows")
print(f"Team stats aggregated: {len(team_stats)} rows")
print(team_stats.head())

# %% Cell 6
def get_advanced_features(df, team_stats_df):
    """Build long-form team performance data with rolling averages and rest days."""
    all_perf = []
    for _, row in df.iterrows():
        all_perf.append({
            'GAME_ID': row['GAME_ID'], 'TEAM_ID': row['HOME_TEAM_ID'],
            'DATE': row['GAME_DATE_EST'], 'OPPONENT_ID': row['VISITOR_TEAM_ID']
        })
        all_perf.append({
            'GAME_ID': row['GAME_ID'], 'TEAM_ID': row['VISITOR_TEAM_ID'],
            'DATE': row['GAME_DATE_EST'], 'OPPONENT_ID': row['HOME_TEAM_ID']
        })

    perf_df = pd.DataFrame(all_perf).merge(team_stats_df, on=['GAME_ID', 'TEAM_ID'])
    perf_df = perf_df.sort_values(['TEAM_ID', 'DATE'])

    # Rest days (capped at 7, shifted to avoid data leakage)
    perf_df['REST_DAYS'] = perf_df.groupby('TEAM_ID')['DATE'].diff().dt.days.fillna(7)
    perf_df['REST_DAYS'] = np.clip(perf_df['REST_DAYS'], 0, 7)

    # Short-term (5) and long-term (15) rolling averages — shifted by 1 to prevent leakage
    stats_cols = ['PTS', 'FG_PCT', 'AST', 'REB']
    for w in [5, 15]:
        rolling = perf_df.groupby('TEAM_ID')[stats_cols].shift(1).rolling(window=w).mean()
        rolling.columns = [f'ROLLING_{c}_{w}' for c in rolling.columns]
        perf_df = pd.concat([perf_df, rolling], axis=1)

    return perf_df


def get_historical_features(df):
    """Compute season win rates and head-to-head (H2H) records per team."""
    df_copy = df.copy()

    teams = []
    for _, row in df_copy.iterrows():
        teams.append({
            'GAME_ID': row['GAME_ID'], 'TEAM_ID': row['HOME_TEAM_ID'],
            'SEASON': row['SEASON'], 'DATE': row['GAME_DATE_EST'], 'WON': row['HOME_TEAM_WINS']
        })
        teams.append({
            'GAME_ID': row['GAME_ID'], 'TEAM_ID': row['VISITOR_TEAM_ID'],
            'SEASON': row['SEASON'], 'DATE': row['GAME_DATE_EST'], 'WON': 1 - row['HOME_TEAM_WINS']
        })

    t_df = pd.DataFrame(teams).sort_values(['TEAM_ID', 'DATE'])
    t_df['SEASON_WIN_RATE'] = t_df.groupby(['TEAM_ID', 'SEASON'])['WON'].shift(1).expanding().mean()

    # H2H win rate over last 3 matchups
    df_copy['MATCHUP'] = df_copy.apply(
        lambda x: tuple(sorted([x['HOME_TEAM_ID'], x['VISITOR_TEAM_ID']])), axis=1
    )
    df_copy['H2H_HOME_WIN_RATE'] = (
        df_copy.groupby('MATCHUP')['HOME_TEAM_WINS'].shift(1).rolling(window=3, min_periods=1).mean()
    )

    h2h = df_copy[['GAME_ID', 'H2H_HOME_WIN_RATE', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']]
    season = t_df[['GAME_ID', 'TEAM_ID', 'SEASON_WIN_RATE']]
    return h2h, season


def get_streaks(df):
    """Compute whether each team won their previous game (momentum indicator)."""
    df = df.sort_values(['TEAM_ID', 'DATE'])
    df['LAST_GAME_WON'] = df.groupby('TEAM_ID')['WON'].shift(1).fillna(0)
    return df[['GAME_ID', 'TEAM_ID', 'LAST_GAME_WON']]


def get_sos(df, season_stats_df):
    """Calculate Strength of Schedule: rolling average of opponents' win rates."""
    opp_stats = season_stats_df[['GAME_ID', 'TEAM_ID', 'SEASON_WIN_RATE']].rename(
        columns={'SEASON_WIN_RATE': 'OPP_WIN_RATE', 'TEAM_ID': 'OPP_TEAM_ID'}
    )
    perf = df.merge(opp_stats, left_on=['GAME_ID', 'OPPONENT_ID'], right_on=['GAME_ID', 'OPP_TEAM_ID'])
    perf = perf.sort_values(['TEAM_ID', 'DATE'])

    sos = perf.groupby('TEAM_ID')['OPP_WIN_RATE'].shift(1).rolling(window=10, min_periods=1).mean()
    perf['SOS_10'] = sos.fillna(0.5)
    return perf[['GAME_ID', 'TEAM_ID', 'SOS_10']]


def get_venue_stats(df):
    """Compute historical home and away win rates per team."""
    h_wr = df.groupby('HOME_TEAM_ID')['HOME_TEAM_WINS'].mean().rename('HIST_HOME_WR')
    v_wr = df.groupby('VISITOR_TEAM_ID')['HOME_TEAM_WINS'].apply(lambda x: (1 - x).mean()).rename('HIST_AWAY_WR')
    return h_wr, v_wr


def calculate_elo(df, k=20):
    """Compute pre-game Elo ratings for each team using a logistic curve."""
    all_teams = pd.concat([df['HOME_TEAM_ID'], df['VISITOR_TEAM_ID']]).unique()
    elo_ratings = {team_id: 1500 for team_id in all_teams}
    elo_history = []

    for _, row in df.sort_values('GAME_DATE_EST').iterrows():
        h_id, v_id = row['HOME_TEAM_ID'], row['VISITOR_TEAM_ID']
        h_elo, v_elo = elo_ratings[h_id], elo_ratings[v_id]

        # Record Elo BEFORE the game starts (no leakage)
        elo_history.append({'GAME_ID': row['GAME_ID'], 'HOME_ELO': h_elo, 'VISITOR_ELO': v_elo})

        exp_h = 1 / (10 ** ((v_elo - h_elo) / 400) + 1)
        act_h = row['HOME_TEAM_WINS']

        elo_ratings[h_id] += k * (act_h - exp_h)
        elo_ratings[v_id] += k * ((1 - act_h) - (1 - exp_h))

    return pd.DataFrame(elo_history)


def weighted_momentum(x):
    """Weighted moving average: last 3 games have 2× weight, prior 7 have 1× weight."""
    weights = np.array([1.0] * 7 + [2.0] * 3)
    if len(x) < 10:
        return np.nan
    return np.dot(x, weights) / weights.sum()


print("Feature engineering functions defined successfully.")

# %% Cell 9
adv_perf = get_advanced_features(games, team_stats)

rolling_cols = [
    'GAME_ID', 'TEAM_ID', 'REST_DAYS',
    'ROLLING_PTS_5', 'ROLLING_FG_PCT_5', 'ROLLING_AST_5', 'ROLLING_REB_5',
    'ROLLING_PTS_15', 'ROLLING_FG_PCT_15', 'ROLLING_AST_15', 'ROLLING_REB_15'
]
games_v3_with_ids = adv_perf[rolling_cols].copy()

games_v4 = games[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 'HOME_TEAM_WINS']].copy()

games_v4 = games_v4.merge(
    games_v3_with_ids, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID']
).rename(
    columns={col: f'HOME_{col}' for col in games_v3_with_ids.columns if col not in ['GAME_ID', 'TEAM_ID']}
).drop(columns=['TEAM_ID'])

games_v4 = games_v4.merge(
    games_v3_with_ids, left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID']
).rename(
    columns={col: f'VISITOR_{col}' for col in games_v3_with_ids.columns if col not in ['GAME_ID', 'TEAM_ID']}
).drop(columns=['TEAM_ID'])

# %% Cell 11
h2h_df, season_stats = get_historical_features(games)

games_v4 = games_v4.merge(h2h_df[['GAME_ID', 'H2H_HOME_WIN_RATE']], on='GAME_ID', how='left')

games_v4 = games_v4.merge(
    season_stats, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left'
).rename(columns={'SEASON_WIN_RATE': 'HOME_SEASON_WIN_RATE'}).drop(columns=['TEAM_ID'])

games_v4 = games_v4.merge(
    season_stats, left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left'
).rename(columns={'SEASON_WIN_RATE': 'VISITOR_SEASON_WIN_RATE'}).drop(columns=['TEAM_ID'])

# %% Cell 13
teams_list = []
for _, row in games.iterrows():
    teams_list.append({
        'GAME_ID': row['GAME_ID'], 'TEAM_ID': row['HOME_TEAM_ID'],
        'DATE': row['GAME_DATE_EST'], 'WON': row['HOME_TEAM_WINS']
    })
    teams_list.append({
        'GAME_ID': row['GAME_ID'], 'TEAM_ID': row['VISITOR_TEAM_ID'],
        'DATE': row['GAME_DATE_EST'], 'WON': 1 - row['HOME_TEAM_WINS']
    })

t_df_long = pd.DataFrame(teams_list)
momentum_df = get_streaks(t_df_long)

games_v5 = games_v4.merge(
    momentum_df, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left'
).rename(columns={'LAST_GAME_WON': 'HOME_LAST_GAME_WON'}).drop(columns=['TEAM_ID'])

games_v5 = games_v5.merge(
    momentum_df, left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left'
).rename(columns={'LAST_GAME_WON': 'VISITOR_LAST_GAME_WON'}).drop(columns=['TEAM_ID'])

games_v5 = games_v5.drop(columns=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'], errors='ignore').fillna(0.5)

print(f"games_v5 shape: {games_v5.shape}")
print(games_v5.head())

# %% Cell 15
sos_df = get_sos(adv_perf[['GAME_ID', 'TEAM_ID', 'DATE', 'OPPONENT_ID']], season_stats)

games_v6 = games[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']].merge(games_v5, left_index=True, right_index=True)

games_v6 = games_v6.merge(
    sos_df, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left'
).rename(columns={'SOS_10': 'HOME_SOS'}).drop(columns=['TEAM_ID'])

games_v6 = games_v6.merge(
    sos_df, left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='left'
).rename(columns={'SOS_10': 'VISITOR_SOS'}).drop(columns=['TEAM_ID'])

games_v6['WIN_RATE_DIFF'] = games_v6['HOME_SEASON_WIN_RATE'] - games_v6['VISITOR_SEASON_WIN_RATE']
games_v6['SOS_DIFF'] = games_v6['HOME_SOS'] - games_v6['VISITOR_SOS']
games_v6 = games_v6.drop(columns=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'], errors='ignore').fillna(0.5)

print(f"games_v6 shape: {games_v6.shape}")
print(games_v6.head())

# %% Cell 17
games_v7 = games_v6.copy()

for stat in ['PTS', 'FG_PCT', 'AST', 'REB']:
    games_v7[f'{stat}_DIFF_5'] = games_v7[f'HOME_ROLLING_{stat}_5'] - games_v7[f'VISITOR_ROLLING_{stat}_5']
    games_v7[f'{stat}_DIFF_15'] = games_v7[f'HOME_ROLLING_{stat}_15'] - games_v7[f'VISITOR_ROLLING_{stat}_15']

h_wr_stats, v_wr_stats = get_venue_stats(games)
games_ids = games[['HOME_TEAM_ID', 'VISITOR_TEAM_ID']]

games_v7 = games_v7.merge(games_ids, left_index=True, right_index=True)
games_v7 = games_v7.merge(h_wr_stats, left_on='HOME_TEAM_ID', right_index=True, how='left')
games_v7 = games_v7.merge(v_wr_stats, left_on='VISITOR_TEAM_ID', right_index=True, how='left')
games_v7 = games_v7.drop(columns=['HOME_TEAM_ID', 'VISITOR_TEAM_ID'], errors='ignore').fillna(0.5)

print(f"games_v7 shape: {games_v7.shape}")
print(games_v7.head())

# %% Cell 19
elo_df = calculate_elo(games)
games_v8_base = games[['GAME_ID']].merge(games_v7, left_index=True, right_index=True)
games_v8 = games_v8_base.merge(elo_df, on='GAME_ID', how='left')

games_v8['REST_DIFF'] = games_v8['HOME_REST_DAYS'] - games_v8['VISITOR_REST_DAYS']
games_v8['ELO_DIFF'] = games_v8['HOME_ELO'] - games_v8['VISITOR_ELO']
games_v8 = games_v8.drop(columns=['GAME_ID'], errors='ignore')

print(f"games_v8 shape: {games_v8.shape}")
print(games_v8.head())

# %% Cell 21
train_data = TabularDataset(games_v8)

predictor_v8 = TabularPredictor(
    label='HOME_TEAM_WINS',
    path='AutoGluonModels_NBA_V8',
    eval_metric='accuracy'
)
predictor_v8.fit(train_data, presets='best_quality', num_stack_levels=2, time_limit=2400)

print("\n--- NBA v8 Leaderboard ---")
print(predictor_v8.leaderboard())

# Feature importance for baseline model
importance_df = predictor_v8.feature_importance(data=train_data, subsample_size=2500, num_shuffle_sets=3)
print("Top 10 features by importance (v8):")
print(importance_df.head(10))

# %% Cell 23
# Drop features with zero or negative importance (add noise, no signal)
low_importance_features = importance_df[importance_df['importance'] <= 0].index.tolist()
games_v8_refined = games_v8.drop(columns=low_importance_features)

print(f"Features dropped ({len(low_importance_features)}): {low_importance_features}")
print(f"Original shape: {games_v8.shape}  →  Refined shape: {games_v8_refined.shape}")
assert 'HOME_TEAM_WINS' in games_v8_refined.columns, "Target column was accidentally dropped!"

train_data_deep = TabularDataset(games_v8_refined)

custom_hyperparameters = {
    'GBM': [
        {'num_leaves': 64,  'learning_rate': 0.05, 'ag_args': {'name_suffix': '_tuned_L1'}},
        {'num_leaves': 128, 'learning_rate': 0.01, 'ag_args': {'name_suffix': '_tuned_L2'}}
    ],
    'NN_TORCH': {
        'num_epochs':    50,
        'hidden_size':   128,
        'dropout_prob':  0.2,
        'learning_rate': 0.001
    }
}

predictor_deep = TabularPredictor(
    label='HOME_TEAM_WINS',
    path='AutoGluonModels_NBA_DeepEnsemble',
    eval_metric='accuracy'
)
predictor_deep.fit(
    train_data=train_data_deep,
    presets='best_quality',
    hyperparameters=custom_hyperparameters,
    num_stack_levels=3,
    num_bag_folds=10,
    time_limit=3600
)

print("\n--- Deep Ensemble Leaderboard ---")
print(predictor_deep.leaderboard())

# %% Cell 25
# Performance metrics
evaluation = predictor_deep.evaluate(train_data_deep)
print("Performance Metrics (Deep Ensemble):")
print(evaluation)

y_true = train_data_deep['HOME_TEAM_WINS']
y_pred = predictor_deep.predict(train_data_deep)
y_prob = predictor_deep.predict_proba(train_data_deep)[1]

cm = confusion_matrix(y_true, y_pred)
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Away Win', 'Home Win'],
    yticklabels=['Away Win', 'Home Win'],
    ax=axes[0]
)
axes[0].set_title('Confusion Matrix: NBA Game Winners')
axes[0].set_xlabel('Predicted Label')
axes[0].set_ylabel('True Label')

axes[1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(loc='lower right')

plt.tight_layout()
plt.show()

# Baseline vs. Deep Ensemble comparison
metrics_v8   = predictor_v8.evaluate(train_data_deep)
metrics_deep = predictor_deep.evaluate(train_data_deep)

acc_v8   = metrics_v8['accuracy']
acc_deep = metrics_deep['accuracy']
abs_improvement = acc_deep - acc_v8
pct_improvement = (abs_improvement / acc_v8) * 100

comparison_df = pd.DataFrame({
    'Metric':           ['Accuracy', 'Log Loss'],
    'Baseline (v8)':   [acc_v8,   metrics_v8.get('log_loss', 'N/A')],
    'Optimized (Deep)': [acc_deep, metrics_deep.get('log_loss', 'N/A')],
    'Improvement (Abs)': [f'+{abs_improvement:.4f}', 'N/A'],
    'Improvement (%)':   [f'{pct_improvement:.2f}%',  'N/A']
})

print("--- Baseline vs. Optimized Model Comparison ---")
print(comparison_df)
print(f"\nDeep Ensemble accuracy: {acc_deep:.2%} ({pct_improvement:.2f}% relative improvement over v8)")

# Feature importance for deep ensemble
importance_deep_df = predictor_deep.feature_importance(
    train_data_deep, num_shuffle_sets=3, subsample_size=2500
)

print("--- Top 10 Features (Deep Ensemble) ---")
print(importance_deep_df.head(10))

plt.figure(figsize=(12, 8))
top_15 = importance_deep_df.head(15)
sns.barplot(x=top_15['importance'], y=top_15.index, hue=top_15.index, palette='viridis', legend=False)
plt.title('Top 15 Most Influential Features: Optimized NBA Predictor')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

for metric in ['ELO_DIFF', 'HOME_ELO', 'SOS_DIFF']:
    if metric in importance_deep_df.index:
        rank  = importance_deep_df.index.get_loc(metric) + 1
        score = importance_deep_df.loc[metric, 'importance']
        print(f"{metric}: Rank {rank}, Score: {score:.4f}")

# %% Cell 28
season_mapping = games[['GAME_ID', 'SEASON']].drop_duplicates()
team_stats_season = team_stats.merge(season_mapping, on='GAME_ID')

stats_to_normalize = ['PTS', 'REB', 'AST']
season_stats_ref = team_stats_season.groupby('SEASON')[stats_to_normalize].agg(['mean', 'std']).reset_index()
season_stats_ref.columns = ['SEASON'] + [
    f'{stat}_{metric}' for stat in stats_to_normalize for metric in ['mean', 'std']
]

team_stats_norm = team_stats_season.merge(season_stats_ref, on='SEASON')
for stat in stats_to_normalize:
    team_stats_norm[f'{stat}_Z'] = (
        (team_stats_norm[stat] - team_stats_norm[f'{stat}_mean']) / team_stats_norm[f'{stat}_std']
    )

team_stats_z_scores = team_stats_norm[['GAME_ID', 'TEAM_ID', 'SEASON', 'PTS_Z', 'REB_Z', 'AST_Z']]

print("League-Relative Z-scores calculated successfully.")
print(season_stats_ref.tail(3))
print(team_stats_z_scores.head())

# %% Cell 30
perf_dates = adv_perf[['GAME_ID', 'TEAM_ID', 'DATE']]
chronological_z = team_stats_z_scores.merge(perf_dates, on=['GAME_ID', 'TEAM_ID'])
chronological_z = chronological_z.sort_values(['TEAM_ID', 'DATE'])

z_cols = ['PTS_Z', 'REB_Z', 'AST_Z']
weighted_features = (
    chronological_z.groupby('TEAM_ID')[z_cols]
    .shift(1)
    .rolling(window=10)
    .apply(weighted_momentum, raw=True)
)
weighted_features = weighted_features.rename(columns={
    'PTS_Z': 'WEIGHTED_MOMENTUM_PTS',
    'REB_Z': 'WEIGHTED_MOMENTUM_REB',
    'AST_Z': 'WEIGHTED_MOMENTUM_AST'
})
chronological_z = pd.concat([chronological_z, weighted_features], axis=1)

print("Weighted Momentum features calculated.")
print(chronological_z.dropna(subset=['WEIGHTED_MOMENTUM_PTS']).head())

# %% Cell 32
chronological_z['GAME_COUNT'] = chronological_z.groupby(['TEAM_ID', 'SEASON']).cumcount()
refined_momentum = chronological_z[chronological_z['GAME_COUNT'] >= 10].copy()

momentum_cols = ['GAME_ID', 'TEAM_ID', 'WEIGHTED_MOMENTUM_PTS', 'WEIGHTED_MOMENTUM_REB', 'WEIGHTED_MOMENTUM_AST']
features_to_merge = refined_momentum[momentum_cols]

games_v9_base = games[['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID']].merge(
    games_v8, left_index=True, right_index=True
)

games_v9_refined = games_v9_base.merge(
    features_to_merge, left_on=['GAME_ID', 'HOME_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='inner'
).rename(columns={
    'WEIGHTED_MOMENTUM_PTS': 'HOME_WEIGHTED_MOMENTUM_PTS',
    'WEIGHTED_MOMENTUM_REB': 'HOME_WEIGHTED_MOMENTUM_REB',
    'WEIGHTED_MOMENTUM_AST': 'HOME_WEIGHTED_MOMENTUM_AST'
}).drop(columns=['TEAM_ID'])

games_v9_refined = games_v9_refined.merge(
    features_to_merge, left_on=['GAME_ID', 'VISITOR_TEAM_ID'], right_on=['GAME_ID', 'TEAM_ID'], how='inner'
).rename(columns={
    'WEIGHTED_MOMENTUM_PTS': 'VISITOR_WEIGHTED_MOMENTUM_PTS',
    'WEIGHTED_MOMENTUM_REB': 'VISITOR_WEIGHTED_MOMENTUM_REB',
    'WEIGHTED_MOMENTUM_AST': 'VISITOR_WEIGHTED_MOMENTUM_AST'
}).drop(columns=['TEAM_ID'])

games_v9_refined = games_v9_refined.drop(
    columns=['GAME_ID', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID'], errors='ignore'
).dropna()

print(f"games_v9_refined shape: {games_v9_refined.shape}")
print(games_v9_refined.head())

# %% Cell 34
custom_hyperparams_v9 = {
    'GBM': [
        {'num_leaves': 64,  'learning_rate': 0.05, 'ag_args': {'name_suffix': '_tuned_L1'}},
        {'num_leaves': 128, 'learning_rate': 0.01, 'ag_args': {'name_suffix': '_tuned_L2'}}
    ],
    'NN_TORCH': {
        'num_epochs':    50,
        'hidden_size':   128,
        'dropout_prob':  0.2,
        'learning_rate': 0.001
    },
    'CAT': {
        'depth':      6,
        'l2_leaf_reg': 3
    }
}

predictor_v9_final = TabularPredictor(
    label='HOME_TEAM_WINS',
    path='AutoGluonModels_NBA_v9_Final',
    eval_metric='accuracy'
)
predictor_v9_final.fit(
    train_data=games_v9_refined,
    presets='best_quality',
    hyperparameters=custom_hyperparams_v9,
    num_stack_levels=4,
    num_bag_folds=10,
    time_limit=3600
)

print("\n--- NBA v9 Final Ensemble Leaderboard ---")
print(predictor_v9_final.leaderboard())

