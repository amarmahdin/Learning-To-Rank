import os
import pandas as pd
import numpy as np
import calendar
import matplotlib.pyplot as plt
import seaborn as sns

def process_and_split_smart_meter(save_output=False):
    file_path = os.path.join('data', 'smart_meter.csv')
    df = pd.read_csv(file_path)

    if 'request_date' not in df.columns or 'meter_code' not in df.columns:
        print("Error: Kolom 'request_date' atau 'meter_code' tidak ditemukan.")
        return

    # Perbaiki format tanggal: ganti 'Z' dengan '+00:00' agar semua bisa diparse
    df['request_date'] = df['request_date'].astype(str).str.replace('Z', '+00:00', regex=False)
    if 'kafka_timestamp' in df.columns:
        df['kafka_timestamp'] = df['kafka_timestamp'].astype(str).str.replace('Z', '+00:00', regex=False)

    # Parsing dengan pd.to_datetime, coba parsing ulang untuk nilai yang masih NaN
    df['request_date'] = pd.to_datetime(df['request_date'], utc=True, errors='coerce')
    nan_request_date = df['request_date'].isna()
    if nan_request_date.any():
        df.loc[nan_request_date, 'request_date'] = pd.to_datetime(
            df.loc[nan_request_date, 'request_date'].astype(str).str.replace('Z', ''), 
            format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce', utc=True
        )

    if 'kafka_timestamp' in df.columns:
        df['kafka_timestamp'] = pd.to_datetime(df['kafka_timestamp'], utc=True, errors='coerce')
        nan_kafka_timestamp = df['kafka_timestamp'].isna()
        if nan_kafka_timestamp.any():
            df.loc[nan_kafka_timestamp, 'kafka_timestamp'] = pd.to_datetime(
                df.loc[nan_kafka_timestamp, 'kafka_timestamp'].astype(str).str.replace('Z', ''), 
                format='%Y-%m-%dT%H:%M:%S.%f', errors='coerce', utc=True
            )

    df['error_code'] = 200
    df['error_message'] = 'Success'

    meter_codes = df['meter_code'].unique()

    sample_date = df['request_date'].iloc[0]
    year, month = sample_date.year, sample_date.month
    days_in_month = calendar.monthrange(year, month)[1]

    total_ideal = len(meter_codes) * 96 * days_in_month
    print(f"Total meter codes: {len(meter_codes)}")
    print(f"Jumlah record: {len(df)}")
    print(f"Total record yang diharapkan: {total_ideal}")
    print(f"Jumlah record yang hilang: {total_ideal - len(df)}")

    new_rows = []

    for meter in meter_codes:
        df_meter = df[df['meter_code'] == meter].sort_values('request_date').reset_index(drop=True)
        min_date = df_meter['request_date'].min().normalize()
        if min_date.tzinfo is not None:
            min_date = min_date.tz_convert('UTC')
        else:
            min_date = min_date.tz_localize('UTC')
        max_date = pd.Timestamp(year=year, month=month, day=days_in_month, tz='UTC')
        # tidak perlu full_range untuk proses ini, tapi tetap bisa dihitung jika ingin validasi

        if save_output:
            end_of_month_timestamp = pd.Timestamp(
                year=year, month=month, day=days_in_month, hour=23, minute=59, second=59, tz='UTC'
            )

            for i in range(len(df_meter) - 1):
                current_row = df_meter.iloc[i]
                next_row = df_meter.iloc[i + 1]

                current_row_dict = current_row.to_dict()
                current_row_dict['generated'] = False
                new_rows.append(current_row_dict)

                diff = (next_row['request_date'] - current_row['request_date']).total_seconds() / 60

                if diff > 15:
                    missing_count = int(diff // 15) - 1
                    for j in range(missing_count):
                        missing_time = current_row['request_date'] + pd.Timedelta(minutes=15 * (j + 1))

                        missing_row = {
                            'meter_code': meter,
                            'request_date': missing_time,
                            'created_at': df_meter.iloc[i]['created_at'] if 'created_at' in df_meter.columns else None,
                            'updated_at': df_meter.iloc[i]['updated_at'] if 'updated_at' in df_meter.columns else None,
                            'error_code': 400,
                            'error_message': 'Failed',
                            'meter_operation_type': current_row['meter_operation_type'] if 'meter_operation_type' in current_row else None,
                            'read_date_time': missing_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                            'source': current_row['source'] if 'source' in current_row else None,
                            'generated': True
                        }

                        if 'kafka_timestamp' in df.columns:
                            if next_row is not None and pd.notna(next_row.get('kafka_timestamp', None)):
                                missing_row['kafka_timestamp'] = next_row['kafka_timestamp']
                            else:
                                missing_row['kafka_timestamp'] = end_of_month_timestamp
                        else:
                            missing_row['kafka_timestamp'] = None

                        # Pastikan semua kolom dari df ada di missing_row
                        for col in df.columns:
                            if col not in missing_row:
                                missing_row[col] = None

                        new_rows.append(missing_row)

            last_row_dict = df_meter.iloc[-1].to_dict()
            last_row_dict['generated'] = False
            new_rows.append(last_row_dict)

            last_date = df_meter['request_date'].max().date()
            end_date = pd.Timestamp(year=year, month=month, day=days_in_month).date()

            if last_date < end_date:
                start_datetime = df_meter['request_date'].max() + pd.Timedelta(minutes=15)
                end_datetime = pd.Timestamp(
                    year=end_date.year, month=end_date.month, day=end_date.day, hour=23, minute=45, tz='UTC'
                )
                missing_times = pd.date_range(start=start_datetime, end=end_datetime, freq='15min')

                for missing_time in missing_times:
                    missing_row = {
                        'meter_code': meter,
                        'request_date': missing_time,
                        'created_at': df_meter.iloc[-1]['created_at'] if 'created_at' in df_meter.columns else None,
                        'updated_at': df_meter.iloc[-1]['updated_at'] if 'updated_at' in df_meter.columns else None,
                        'error_code': 400,
                        'error_message': 'Failed',
                        'meter_operation_type': df_meter.iloc[-1]['meter_operation_type'] if 'meter_operation_type' in df_meter.columns else None,
                        'read_date_time': missing_time.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                        'source': df_meter.iloc[-1]['source'] if 'source' in df_meter.columns else None,
                        'generated': True
                    }

                    if 'kafka_timestamp' in df.columns:
                        missing_row['kafka_timestamp'] = end_of_month_timestamp
                    else:
                        missing_row['kafka_timestamp'] = None

                    # Pastikan semua kolom dari df ada di missing_row
                    for col in df.columns:
                        if col not in missing_row:
                            missing_row[col] = None

                    new_rows.append(missing_row)

    if save_output:
        df_new = pd.DataFrame(new_rows)
        # jadikan seluruh kafka_timestamp dalam format waktu
        if 'kafka_timestamp' in df_new.columns:
            df_new['kafka_timestamp'] = pd.to_datetime(df_new['kafka_timestamp'], utc=True, errors='coerce')

        df_new = df_new.sort_values(['meter_code', 'request_date']).reset_index(drop=True)

        output_dir = os.path.join('result', 'dataset')
        os.makedirs(output_dir, exist_ok=True)

        assigned_devices_file = os.path.join('data', 'assigned_devices.csv')
        df_assigned = pd.read_csv(assigned_devices_file)
        df_assigned.columns = [col.strip() for col in df_assigned.columns]

        assigned_tech_col = next((col for col in df_assigned.columns if 'Assigned Technology' == col), None)
        if assigned_tech_col is None:
            print("Kolom 'Assigned Technology' tidak ditemukan di assigned_devices.csv")
            return

        mapping = dict(zip(df_assigned['ID'], df_assigned[assigned_tech_col]))
        df_new['assigned_technology'] = df_new['meter_code'].map(mapping)

        combined_output_path = os.path.join(output_dir, 'processed_smart_meter_with_technology.csv')
        df_new.to_csv(combined_output_path, index=False)
        print(f"Data lengkap dengan assigned_technology disimpan di {combined_output_path}")

        loss_summary = df_new[df_new['error_code'] == 400].groupby('assigned_technology').size().reset_index(name='loss_count')
        total_summary = df_new.groupby('assigned_technology').size().reset_index(name='total_count')
        summary = loss_summary.merge(total_summary, on='assigned_technology', how='right').fillna(0)
        summary['loss_percentage'] = (summary['loss_count'] / summary['total_count']) * 100

        summary_path = os.path.join('result', 'metrics', 'loss_demographics.csv')
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        summary.to_csv(summary_path, index=False)
        print(f"Loss demographics saved to {summary_path}")

        plt.figure(figsize=(10,6))
        sns.barplot(data=summary, x='assigned_technology', y='loss_percentage')
        plt.title('Data Loss Percentage per Technology')
        plt.ylabel('Loss Percentage (%)')
        plt.xlabel('Technology')
        plt.xticks(rotation=45)
        plt.tight_layout()

        image_path = os.path.join('result', 'image', 'loss_percentage_per_technology.png')
        os.makedirs(os.path.dirname(image_path), exist_ok=True)
        plt.savefig(image_path)
        plt.close()
        print(f"Loss percentage plot saved to {image_path}")

if __name__ == '__main__':
    process_and_split_smart_meter(save_output=True)