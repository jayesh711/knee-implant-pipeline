import fitz
import sys

doc = fitz.open(r'C:\Users\jdongre\Downloads\Sub-Millimeter Knee Segmentation for Robotics.pdf')
text = ''
for page in doc:
    text += page.get_text()

# Write to file to avoid encoding issues
with open(r'd:\knee-implant-pipeline\knee-implant-pipeline\scripts\pdf_output.txt', 'w', encoding='utf-8') as f:
    f.write(text)

print("Done - wrote", len(text), "chars")
