import pandas as pd
import os


#  Load AI4I 2020 Dataset

ai4i_path = os.path.join("datasets", "ai4i2020.csv")
ai4i = pd.read_csv(ai4i_path)

# Select relevant sensor/operational columns
ai4i_features = ai4i[['Air temperature [K]', 'Process temperature [K]',
                      'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']].copy()

# Map faults: Electrical = PWF/HDF, Operational = TWF/OSF, Normal = 0
def map_ai4i_fault(row):
    if row['Machine failure'] == 'No Failure':
        return 0  # Normal
    elif row.get('PWF', 0) == 1 or row.get('HDF', 0) == 1:
        return 2  # Electrical Fault
    elif row.get('TWF', 0) == 1 or row.get('OSF', 0) == 1:
        return 4  # Operational Fault
    else:
        return 0  # Default Normal

ai4i_labels = ai4i.apply(map_ai4i_fault, axis=1)


#  Load Robot Execution Dataset

robot_files = ["LP1.data", "LP2.data", "LP3.data", "LP4.data", "LP5.data"]
robot_data_list = []

for f in robot_files:
    file_path = os.path.join("datasets", f)
    with open(file_path, 'r') as file:
        lines = file.readlines()

    cleaned_rows = []
    for line in lines:
        line = line.strip()
        if line == "":
            continue
        parts = line.split()
        if len(parts) != 6:
            continue  # skip malformed lines
        cleaned_rows.append([float(x) for x in parts])

    df = pd.DataFrame(cleaned_rows, columns=['Fx','Fy','Fz','Tx','Ty','Tz'])
    robot_data_list.append(df)

robot_data = pd.concat(robot_data_list, ignore_index=True)

# Label Mechanical (1) and Sensory (3) faults
total_rows = robot_data.shape[0]
half = total_rows // 2
robot_data['Fault_Label'] = [1]*half + [3]*(total_rows - half)

robot_features = robot_data[['Fx','Fy','Fz','Tx','Ty','Tz']].copy()
robot_labels = robot_data['Fault_Label']

# Align Columns for Merge
# Add missing columns using .loc to avoid SettingWithCopyWarning
for col in ['Fx','Fy','Fz','Tx','Ty','Tz']:
    ai4i_features.loc[:, col] = 0

for col in ['Air temperature [K]', 'Process temperature [K]',
            'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']:
    robot_features.loc[:, col] = 0


#  Merge Features and Labels

merged_features = pd.concat([ai4i_features, robot_features], ignore_index=True)
merged_labels = pd.concat([ai4i_labels, robot_labels], ignore_index=True)

merged_dataset = merged_features.copy()
merged_dataset['Fault_Label'] = merged_labels

#  Save Master Dataset
output_path = os.path.join("datasets", "robot_fault_dataset.csv")
merged_dataset.to_csv(output_path, index=False)
print(f"✅ Master dataset created successfully! Shape: {merged_dataset.shape}")
print(f"Saved at: {output_path}")
