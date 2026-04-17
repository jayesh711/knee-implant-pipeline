import SimpleITK as sitk
import os
import numpy as np

def analyze_dicom_dir(path):
    print(f"Analyzing: {path}")
    files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dcm')]
    print(f"Total files: {len(files)}")
    
    series_info = {}
    for f in files:
        reader = sitk.ImageFileReader()
        reader.SetFileName(f)
        reader.ReadImageInformation()
        s_id = reader.GetMetaData('0020|000e')
        s_desc = reader.GetMetaData('0008|103e') if reader.HasMetaDataKey('0008|103e') else 'None'
        pos_str = reader.GetMetaData('0020|0032')
        pos = [float(x) for x in pos_str.split('\\')]
        
        if s_id not in series_info:
            series_info[s_id] = {'desc': s_desc, 'positions': []}
        series_info[s_id]['positions'].append(pos)
    
    print(f"\nFound {len(series_info)} distinct series:")
    for s_id, info in series_info.items():
        pos_arr = np.array(info['positions'])
        z_coords = np.sort(pos_arr[:, 2])
        print(f"  Series {s_id}:")
        print(f"    Description: {info['desc']}")
        print(f"    Count:       {len(info['positions'])}")
        print(f"    Z-Range:     {z_coords[0]} to {z_coords[-1]} ({z_coords[-1]-z_coords[0]:.1f} mm)")
    
    # Check for duplicates
    if len(np.unique(z_coords)) < len(z_coords):
        print(f"WARNING: Found {len(z_coords) - len(np.unique(z_coords))} duplicate Z positions!")

if __name__ == "__main__":
    analyze_dicom_dir(r"D:\knee-implant-pipeline\knee-implant-pipeline\data\DICOM\AB72Y_Left")
