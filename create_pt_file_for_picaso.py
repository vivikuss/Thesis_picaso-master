import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Paths to folder with results and PICASO base cases folder
### ADJUST THIS PART ###
res_folder_path = '/Users/new/Desktop/THESIS'
output_folder_path = '/Users/new/Desktop/THESIS/THESIS_picaso-master/reference/base_cases'

# Folder names of results and model names
### ADJUST THIS PART ###
# K2-18b
#file_labels = ['results_k218b_hybrid_mdwarf_he_99p', 'results_k218b_hybrid_mdwarf_he_50p', 'results_k218b_hybrid_mdwarf_h2_99p']
# LHS 1140b
file_labels = ['results_lhs_hybrid_mdwarf_he_99p', 'results_lhs_hybrid_mdwarf_he_50p', 'results_lhs_hybrid_mdwarf_h2_99p']
model_labels = ['model1p.pt','model50p.pt','modelh2.pt']
gases = ['He', 'He', 'H2']

# Create pt files for PICASO
for file, model, gas in zip(file_labels,model_labels,gases):
    flay_path = res_folder_path + '/' + file + '/flay.out'
    play_path = res_folder_path + '/' + file + '/play.out'
    Tlay_path = res_folder_path + '/' + file + '/Tlay.out'
    output_path = output_folder_path + '/' + model

    # Load the data 
    flay_raw = np.loadtxt(flay_path, skiprows=1)  # 200 rows, 4 per layer
    play_data = np.loadtxt(play_path, skiprows=1)  
    Tlay_data = np.loadtxt(Tlay_path, skiprows=1) 

    # Convert pressure from Pa to bar 
    play_data_bar = play_data/1e5

    # Reshape flay_data (200 rows -> 50 layers, each with 4 columns: He, CO2, CH4, H2O)
    flay_data = flay_raw.reshape(50, 4) 

    # Stack everything into a single array and reverse order
    pt_data = np.column_stack((flay_data, play_data_bar, Tlay_data))[::-1]

    # Create a DataFrame
    columns = [str(gas), "CO2", "CH4", "H2O", "pressure", "temperature"] #in bar, in K
    pt_df = pd.DataFrame(pt_data, columns=columns)

    # Save to file 
    pt_df.to_csv(output_path, sep=' ', index=False)

    print(f"File '{model}' successfully created!")

