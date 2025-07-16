import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random

from src.ranknet import RankNet
from src.lambdarank import LambdaRank
from src.evaluator import RankingEvaluator
from src.process_and_split_smart_meter import process_and_split_smart_meter

def stratified_train_val_test_split(df, group_col, train_frac=0.7, val_frac=0.2, random_state=42):
    meter_codes = df[group_col].unique()
    random.seed(random_state)
    meter_codes = list(meter_codes)
    random.shuffle(meter_codes)
    n = len(meter_codes)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)
    train_codes = meter_codes[:n_train]
    val_codes = meter_codes[n_train:n_train+n_val]
    test_codes = meter_codes[n_train+n_val:]
    df_train = df[df[group_col].isin(train_codes)].reset_index(drop=True)
    df_val = df[df[group_col].isin(val_codes)].reset_index(drop=True)
    df_test = df[df[group_col].isin(test_codes)].reset_index(drop=True)
    return df_train, df_val, df_test

class SimplePairwiseDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, x_i, x_j, labels, batch_size=64):
        self.x_i = x_i
        self.x_j = x_j
        self.labels = labels
        self.batch_size = batch_size
        self.n = x_i.shape[0]
        self.indices = np.arange(self.n)
    def __len__(self): return int(np.ceil(self.n / self.batch_size))
    def __getitem__(self, idx):
        batch_idx = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        X = np.stack([self.x_i[batch_idx], self.x_j[batch_idx]], axis=1)
        y = self.labels[batch_idx]
        return X, y

def add_extra_features(df):
    df['request_date'] = pd.to_datetime(df['request_date'], errors='coerce', utc=True)
    df['hour'] = df['request_date'].dt.hour
    df['day_of_week'] = df['request_date'].dt.weekday
    df['is_weekend'] = ((df['day_of_week'] >= 5) * 1).astype(int)
    # Fitur polynomial/interaction waktu
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
    df['is_rush_hour'] = ((df['hour'].isin([7,8,9,17,18,19]))).astype(int)
    df['night_rush_interaction'] = df['is_night'] * df['is_rush_hour']
    df['night_weekend_interaction'] = df['is_night'] * df['is_weekend']
    df['rush_weekend_interaction'] = df['is_rush_hour'] * df['is_weekend']
    # Rolling mean delay per meter_code (window 4, min_periods=1)
    df = df.sort_values(['meter_code', 'request_date'])
    df['rolling_mean_delay'] = df.groupby('meter_code')['delay'].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    df['rolling_std_delay'] = df.groupby('meter_code')['delay'].transform(lambda x: x.rolling(window=4, min_periods=1).std().fillna(0))
    df['exp_mean_delay'] = df.groupby('meter_code')['delay'].transform(lambda x: x.ewm(span=4, min_periods=1).mean())
    df['mean_delay_meter'] = df.groupby('meter_code')['delay'].transform('mean')
    # --- Tempat menambah fitur domain/temporal lain ---
    # Contoh: df['feature_baru'] = ...
    return df

def plot_delay_distribution_per_technology(df, save_dir='result/image'):
    os.makedirs(save_dir, exist_ok=True)
    techs = df['assigned_technology'].unique()
    for tech in techs:
        plt.figure(figsize=(8,4))
        subset = df[df['assigned_technology'] == tech]
        plt.hist(subset['delay_norm_global'], bins=50, color='skyblue', edgecolor='black')
        plt.title(f'Distribusi delay_norm_global: {tech}')
        plt.xlabel('delay_norm_global')
        plt.ylabel('Jumlah Data')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'distribusi_delay_norm_{tech}.png'))
        plt.close()
        print(f"[INFO] Histogram delay_norm_global untuk teknologi {tech} disimpan ke {save_dir}/distribusi_delay_norm_{tech}.png")

def create_pairwise_samples(df, pair_per_sample=10, delay_threshold=0.0, random_state=42, feature_cols=None):
    """
    Hanya buat pasangan i-j jika abs(delay_norm_global_i - delay_norm_global_j) > delay_threshold.
    """
    np.random.seed(random_state)
    features = df[feature_cols].values
    target = df['delay_norm_global'].values
    n = len(features)
    xi_list, xj_list, label_list = [], [], []
    for i in range(n):
        if i % 5000 == 0:
            print(f"Pairwise progress: {i}/{n}")
        # Pilih kandidat dengan beda delay signifikan saja
        candidates = [j for j in range(n) if j != i and abs(target[i] - target[j]) > delay_threshold]
        if len(candidates) == 0: continue
        sampled_js = np.random.choice(candidates, min(pair_per_sample, len(candidates)), replace=False)
        for j in sampled_js:
            xi = features[i]
            xj = features[j]
            label = 1 if target[i] < target[j] else 0
            xi_list.append(xi)
            xj_list.append(xj)
            label_list.append(label)
            xi_list.append(xj)
            xj_list.append(xi)
            label_list.append(1-label)
    x_i = np.array(xi_list, dtype=np.float32)
    x_j = np.array(xj_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    print(f"Pairwise samples created: {len(labels)} (hanya pair beda delay > {delay_threshold})")
    print("Distribusi label pairwise:", np.unique(labels, return_counts=True))
    return x_i, x_j, labels

def plot_ndcg_and_save(ndcg_scores_per_tech, tech_list, model_name, postfix='validation', k=10):
    color_map = {
        'Cellular': 'purple', 'BBPLC': 'goldenrod',
        'NBPLC': 'orange', 'Sigfox': 'brown', 'LoRa': 'yellowgreen'
    }
    tech_score_pairs = sorted(zip(tech_list, ndcg_scores_per_tech), key=lambda x: (0 if np.isnan(x[1]) else x[1]), reverse=True)
    sorted_labels = [tech for tech, _ in tech_score_pairs]
    sorted_scores = [score for _, score in tech_score_pairs]
    sorted_colors = [color_map.get(label, 'blue') for label in sorted_labels]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_scores)), sorted_scores, color=sorted_colors)
    plt.title(f'{model_name} NDCG@{k} Curve ({postfix.upper()})')
    plt.xlabel(f'NDCG@{k}')
    plt.ylabel('Technology')
    plt.xlim(0, 1.0)
    ax = plt.gca()
    ax.set_yticks(range(len(sorted_labels)))
    ax.set_yticklabels(sorted_labels)
    plt.grid(True, axis='x', linestyle='--', alpha=0.7)
    plt.gca().invert_yaxis()
    os.makedirs('result/image', exist_ok=True)
    plt.savefig(f'result/image/{model_name.lower()}_ndcg_curve_{postfix}_k{k}.png')
    plt.close()
    df_rank = pd.DataFrame({
        'No': range(1, len(sorted_labels) + 1),
        'Teknologi Telekomunikasi': sorted_labels,
        f'{model_name} Score': sorted_scores,
        'Rank': range(1, len(sorted_labels) + 1)
    })
    os.makedirs('result/metrics', exist_ok=True)
    df_rank.to_csv(f'result/metrics/{model_name.lower()}_ranking_results_{postfix}_k{k}.csv', index=False)
    print(f"Hasil grafik dan peringkat {model_name} ({postfix}) NDCG@{k} disimpan\n")

def train_model(model, train_gen, val_gen, model_name="RankNet", epochs=15):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=getattr(model, 'pairwise_loss', getattr(model, 'lambda_loss', None))
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    history = model.fit(
        train_gen, epochs=epochs, steps_per_epoch=len(train_gen),
        validation_data=val_gen, validation_steps=len(val_gen),
        callbacks=[early_stopping], verbose=1
    )
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"{model_name} Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    os.makedirs('result/image', exist_ok=True)
    plt.savefig(f"result/image/{model_name.lower()}_loss.png")
    plt.close()
    return model

def evaluate_instance_ranking_all_k(model, df_eval, feature_cols, model_name="RankNet", postfix="validation"):
    X_eval = df_eval[feature_cols].values
    scores = model.scoring(X_eval).numpy().flatten()
    df_eval = df_eval.copy()
    df_eval['score'] = scores
    print(f"\n[DEBUG] Distribusi nilai delay_norm_global pada {postfix}:")
    print(df_eval['delay_norm_global'].describe())
    print(f"[DEBUG] Distribusi assigned_technology pada {postfix}:")
    print(df_eval['assigned_technology'].value_counts())
    for k in [10, 20, 50]:
        ndcgs, mean_ndcg = RankingEvaluator(k=k).evaluate_instance_ranking(
            df_eval, label_col='delay_norm_global', pred_col='score', group_col='assigned_technology'
        )
        print(f"\nNDCG@{k} per teknologi untuk {model_name} ({postfix}):")
        for tech, ndcg in ndcgs.items():
            print(f"  {tech:10s}: {ndcg:.4f}")
        print(f"\nNDCG@{k} keseluruhan {model_name} ({postfix}): {mean_ndcg:.4f}")
        tech_list = [t for t in ['Cellular', 'BBPLC', 'NBPLC', 'Sigfox', 'LoRa'] if t in ndcgs]
        ndcg_scores_per_tech = [ndcgs.get(t, np.nan) for t in tech_list]
        plot_ndcg_and_save(ndcg_scores_per_tech, tech_list, model_name, postfix=postfix, k=k)
    # Tambahan untuk NDCG@all
    ndcgs_all, mean_ndcg_all = RankingEvaluator(k=None).evaluate_instance_ranking(
        df_eval, label_col='delay_norm_global', pred_col='score', group_col='assigned_technology'
    )
    print(f"\nNDCG@all per teknologi untuk {model_name} ({postfix}):")
    for tech, ndcg in ndcgs_all.items():
        print(f"  {tech:10s}: {ndcg:.4f}")
    print(f"\nNDCG@all keseluruhan {model_name} ({postfix}): {mean_ndcg_all:.4f}")
    tech_list = [t for t in ['Cellular', 'BBPLC', 'NBPLC', 'Sigfox', 'LoRa'] if t in ndcgs_all]
    ndcg_scores_per_tech = [ndcgs_all.get(t, np.nan) for t in tech_list]
    plot_ndcg_and_save(ndcg_scores_per_tech, tech_list, model_name, postfix=postfix, k='all')

def save_delay_per_technology(df, postfix='test'):
    os.makedirs('result/metrics', exist_ok=True)
    delay_count = df.groupby('assigned_technology').size()
    delay_sum = df.groupby('assigned_technology')['delay'].sum().astype(int)
    delay_stats = pd.DataFrame({
        'Jumlah Data': delay_count,
        'Total Delay (detik)': delay_sum
    })
    print(f"\n[DEBUG] Delay per teknologi pada {postfix}:")
    for tech, row in delay_stats.iterrows():
        print(f"{tech:10s} {row['Jumlah Data']:4.0f} : {row['Total Delay (detik)']}")
    delay_stats.to_csv(f'result/metrics/delay_count_per_technology_{postfix}.csv')
    print(f"Delay per teknologi {postfix} disimpan ke result/metrics/delay_count_per_technology_{postfix}.csv\n")

def main():
    data_path = 'result/dataset/processed_smart_meter_with_technology.csv'

    if not os.path.exists(data_path):
        print(f"{data_path} belum ditemukan. Memproses data mentah dulu ...")
        process_and_split_smart_meter(save_output=True)
        if not os.path.exists(data_path):
            print("Gagal membuat file data hasil preprocessing. Cek data mentah Anda!")
            exit(1)
        else:
            print(f"File hasil preprocessing {data_path} berhasil dibuat.\n")

    df = pd.read_csv(data_path)

    # Pastikan kolom delay sudah ada
    if 'delay' not in df.columns:
        df['request_date'] = pd.to_datetime(df['request_date'], errors='coerce', utc=True)
        df['kafka_timestamp'] = pd.to_datetime(df['kafka_timestamp'], errors='coerce', utc=True)
        df['delay'] = (df['kafka_timestamp'] - df['request_date']).dt.total_seconds().abs()

    if not all(col in df.columns for col in ['delay_norm_global', 'avg_delay_norm', 'missing_ratio', 'missing_flag']):
        print("Membuat fitur delay_norm_global, avg_delay_norm, missing_ratio, missing_flag ...")
        df['request_date'] = pd.to_datetime(df['request_date'], errors='coerce', utc=True)
        df['kafka_timestamp'] = pd.to_datetime(df['kafka_timestamp'], errors='coerce', utc=True)
        df['delay'] = (df['kafka_timestamp'] - df['request_date']).dt.total_seconds().abs()
        global_min_delay = df['delay'].min()
        global_max_delay = df['delay'].max()
        df['delay_norm_global'] = (df['delay'] - global_min_delay) / (global_max_delay - global_min_delay)
        df['delay_norm_global'] = df['delay_norm_global'].fillna(0.0)
        if 'missing_flag' not in df.columns:
            df['missing_flag'] = ((df.get('error_code', 0) == 400) | (df.get('generated', False) == True)).astype(int)
        agg_stats = df.groupby('assigned_technology').agg(
            avg_delay=('delay', 'mean'),
            missing_count=('missing_flag', 'sum'),
            total_count=('delay', 'count')
        ).reset_index()
        agg_stats['missing_ratio'] = agg_stats['missing_count'] / agg_stats['total_count']
        df = df.merge(
            agg_stats[['assigned_technology', 'avg_delay', 'missing_ratio']],
            on='assigned_technology', how='left'
        )
        df['avg_delay_norm'] = (df['avg_delay'] - global_min_delay) / (global_max_delay - global_min_delay)
        df['avg_delay_norm'] = df['avg_delay_norm'].fillna(0.0)
        df['missing_ratio'] = df['missing_ratio'].fillna(0.0)

    df = add_extra_features(df)

    # Plot distribusi delay_norm_global per teknologi
    plot_delay_distribution_per_technology(df)

    feature_cols = [
        'delay_norm_global', 'avg_delay_norm', 'missing_ratio', 'missing_flag',
        'hour', 'day_of_week', 'is_weekend',
        'is_night', 'is_rush_hour', 'night_rush_interaction', 'night_weekend_interaction', 'rush_weekend_interaction',
        'rolling_mean_delay', 'rolling_std_delay', 'exp_mean_delay', 'mean_delay_meter'
        # Jika ingin menambah fitur domain, tambahkan di sini
    ]

    df_train, df_val, df_test = stratified_train_val_test_split(df, group_col="meter_code")
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    save_delay_per_technology(df_test, postfix='test')

    # Hanya buat pairwise dengan beda delay_norm_global signifikan
    x_i_train, x_j_train, y_train = create_pairwise_samples(df_train, pair_per_sample=10, delay_threshold=0.02, feature_cols=feature_cols)
    x_i_val, x_j_val, y_val = create_pairwise_samples(df_val, pair_per_sample=10, delay_threshold=0.02, feature_cols=feature_cols)
    train_gen = SimplePairwiseDataGenerator(x_i_train, x_j_train, y_train, batch_size=32)
    val_gen = SimplePairwiseDataGenerator(x_i_val, x_j_val, y_val, batch_size=32)

    input_dim = x_i_train.shape[1]
    ranknet = RankNet(input_dim=input_dim, hidden_layers=[32, 16], dropout_rate=0.1, l2_reg=0.001)
    lambdarank = LambdaRank(input_dim=input_dim, hidden_layers=[32, 16], dropout_rate=0.1, l2_reg=0.001)

    print("\nTraining Ranknet")
    ranknet = train_model(ranknet, train_gen, val_gen, model_name="RankNet", epochs=20)
    print("\nTraining Lambdarank")
    lambdarank = train_model(lambdarank, train_gen, val_gen, model_name="LambdaRank", epochs=20)

    print("\n=== EVALUASI RANKING INSTANCE PADA DATA VALIDASI ===")
    evaluate_instance_ranking_all_k(ranknet, df_val, feature_cols, model_name="RankNet", postfix="validation")
    evaluate_instance_ranking_all_k(lambdarank, df_val, feature_cols, model_name="LambdaRank", postfix="validation")

    print("\n=== EVALUASI RANKING INSTANCE PADA DATA TEST ===")
    evaluate_instance_ranking_all_k(ranknet, df_test, feature_cols, model_name="RankNet", postfix="test")
    evaluate_instance_ranking_all_k(lambdarank, df_test, feature_cols, model_name="LambdaRank", postfix="test")

    os.makedirs('result/models', exist_ok=True)
    ranknet.save('result/models/ranknet_model_val.h5')
    lambdarank.save('result/models/lambdarank_model_val.h5')
    print("Model tersimpan di result/models/")

if __name__ == "__main__":
    main()