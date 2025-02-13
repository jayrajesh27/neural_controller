import pandas as pd

# Converted data into nanosecond timestamp conversion

cmd_vel = pd.read_csv('diff_drive_cmd_vel.csv')
odom = pd.read_csv('diff_drive_odometry.csv')

cmd_vel['timestamp'] = pd.to_datetime(cmd_vel['timestamp'], unit='ns')
odom['timestamp'] = pd.to_datetime(odom['timestamp'], unit='ns')

# time-indexed DataFrames

cmd_vel = cmd_vel.set_index('timestamp').sort_index()
odom = odom.set_index('timestamp').sort_index()

# Resampled cmd_vel file into forward-fill at odometry's native frequency
resampled_vel = cmd_vel.resample('1ms').ffill().reindex(odom.index, method='ffill')

merged = odom.join(resampled_vel, how='left', rsuffix='_cmd')

# initial values before first command

merged[['linear_x', 'angular_z']] = merged[['linear_x', 'angular_z']].fillna(0)

final_columns = {
    'pose_x': 'pose_x',
    'pose_y': 'pose_y',
    'orientation_y': 'orientation_y',
    'orientation_w': 'orientation_w',
    'linear_velocity_x': 'odom_linear_x',
    'angular_velocity_z': 'odom_angular_z',
    'linear_x': 'cmd_linear_x',
    'angular_z': 'cmd_angular_z'
}

merged[final_columns.keys()].rename(columns=final_columns).to_csv('merged_navigation_data.csv')
