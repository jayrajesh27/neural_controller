import pandas as pd
import numpy as np
import os

def process_lidar_ranges(df):
    """Split and clean LIDAR scan ranges column"""
    lidar_split = df['ranges'].str.split(',', expand=True)
    lidar_split = lidar_split.apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan)
    return lidar_split.dropna(axis=1, how='all')

data_dir = '/Users/jayrajesh/Downloads/OneDrive_1_2-19-2025'
output_dir = os.path.join(data_dir, 'processed')
os.makedirs(output_dir, exist_ok=True)

try:
    cmd_vel = pd.read_csv(os.path.join(data_dir, 'diff_drive_cmd_vel.csv'))
    odom = pd.read_csv(os.path.join(data_dir, 'diff_drive_odometry.csv'))
    lidar = pd.read_csv(os.path.join(data_dir, 'diff_drive_scan.csv'))

    for df in [cmd_vel, odom, lidar]:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', errors='coerce')
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)

    # Processed LIDAR ranges
    lidar_clean = process_lidar_ranges(lidar)
    lidar_clean.columns = [f'range_{i}' for i in range(1, len(lidar_clean.columns)+1)]

    # Merged using nearest timestamps
    merged = pd.merge_asof(
        odom.sort_index(),
        lidar_clean.sort_index(),
        left_index=True,
        right_index=True,
        direction='nearest',
        tolerance=pd.Timedelta('50ms') 
    )

    # Merge with command velocities
    resampled_vel = cmd_vel.resample('1ms').ffill()
    merged = pd.merge_asof(
        merged,
        resampled_vel,
        left_index=True,
        right_index=True,
        direction='nearest',
        tolerance=pd.Timedelta('50ms')
    )

    merged[['linear_x', 'angular_z']] = merged[['linear_x', 'angular_z']].fillna(0)
    
    output_columns = {
        'pose_x': 'pose_x',
        'pose_y': 'pose_y',
        'orientation_y': 'orientation_y',
        'orientation_w': 'orientation_w',
        'linear_velocity_x': 'odom_linear_x',
        'angular_velocity_z': 'odom_angular_z',
        'linear_x': 'cmd_linear_x',
        'angular_z': 'cmd_angular_z'
    }
    
    output_path = os.path.join(output_dir, 'merged_navigation_data.csv')
    merged.rename(columns=output_columns)[list(output_columns.values()) + list(lidar_clean.columns)].to_csv(output_path)

except FileNotFoundError as e:
    print(f"Missing required file: {e.filename}")
except KeyError as e:
    print(f"Missing required column: {e}")





