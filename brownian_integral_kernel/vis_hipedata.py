import zipfile
from os.path import exists
import pandas as pd
from matplotlib import pyplot as plt

base_path="./brownian_integral_kernel/HIPE"
zip_file = "hipe_cleaned_v1.0.1_geq_2017-10-23_lt_2017-10-30.zip"

if not exists(base_path+"/ChipPress_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv"):
    with zipfile.ZipFile(base_path+"/"+zip_file, 'r') as zip_ref:
        zip_ref.extractall(base_path)

datasets = [
    "ChipPress_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
    "ChipSaw_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
    "HighTemperatureOven_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
    "MainTerminal_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
    "PickAndPlaceUnit_PhaseCount_2_geq_2017-10-23_lt_2017-10-30.csv",
    "ScreenPrinter_PhaseCount_2_geq_2017-10-23_lt_2017-10-30.csv",
    "SolderingOven_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
    "VacuumOven_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
    "VacuumPump1_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
    "VacuumPump2_PhaseCount_2_geq_2017-10-23_lt_2017-10-30.csv",
    "WashingMachine_PhaseCount_3_geq_2017-10-23_lt_2017-10-30.csv",
]


df = pd.read_csv(base_path+"/"+datasets[1])
df=df.head(1000)
df['Time'] = pd.to_datetime(df['SensorDateTime'],format='ISO8601')
print(df.columns)
for c in df.columns:
    print(c)
    print(df[c])
    plt.plot(df['Time'].astype(int)-df['Time'].astype(int)[0],df[c])
    plt.show()