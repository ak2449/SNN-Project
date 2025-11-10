import numpy as np
import snntorch as snn
import h5py

file_path = 'radar_training_dataset.mat'

try:
    with h5py.File(file_path, 'r') as f:
        print("Variables in file:")
        print(list(f.keys()))

        data_h5 = f['dataset_data']
        label_r_h5 = f['dataset_labels_range']
        label_v_h5 = f['dataset_labels_velocity']


        dataset_data = data_h5[()]
        dataset_labels_range = label_r_h5[()]
        dataset_labels_velocity = label_v_h5[()]
        
        print(f"\nShape of 'dataset_data': {dataset_data.shape}")
        print(f"Data type: {dataset_data.dtype}")
        
        print(f"\nShape of 'dataset_labels_range': {dataset_labels_range.shape}")
        print(f"\nShape of 'dataset_labels_velocity': {dataset_labels_velocity.shape}")
        

        if np.iscomplexobj(dataset_data):
             print("\nData is complex (as expected).")
        # If it loads as a struct (common issue):
        elif 'real' in dataset_data.dtype.names:
             print("\nData loaded as struct. Recombining to complex...")
             dataset_data = dataset_data['real'] + 1j * dataset_data['imag']
             print(f"New complex shape: {dataset_data.shape}")
             print(f"New Data type: {dataset_data.dtype}")


except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}")