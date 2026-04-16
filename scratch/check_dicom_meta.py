
import pydicom
import os
from pathlib import Path

def check_dicom_metadata(patient_id):
    dicom_dir = Path("data/DICOM") / patient_id
    if not dicom_dir.exists():
        print(f"Directory {dicom_dir} not found")
        return
    
    files = [f for f in dicom_dir.iterdir() if f.is_file()]
    if not files:
        print(f"No files found in {dicom_dir}")
        return
    
    # Check first file
    ds = pydicom.dcmread(str(files[0]))
    
    tags = [
        "StudyDescription",
        "ProtocolName",
        "ImageComments",
        "RequestedProcedureDescription",
        "AdmittingDiagnosesDescription",
        "SeriesDescription",
        "PatientComments"
    ]
    
    print(f"--- Metadata for {patient_id} ---")
    for tag in tags:
        val = getattr(ds, tag, "N/A")
        print(f"{tag}: {val}")
    
    # Check if there are multiple series
    series = set()
    for f in files[:100]: # Check first 100 files for series variety
        try:
            d = pydicom.dcmread(str(f), stop_before_pixels=True)
            series.add(d.SeriesDescription if hasattr(d, 'SeriesDescription') else "Unknown")
        except:
            pass
    print(f"Unique Series found in first 100 files: {series}")

if __name__ == "__main__":
    check_dicom_metadata("S0001")
