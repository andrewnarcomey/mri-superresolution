import zipfile

with zipfile.ZipFile('processed_data.zip', 'r') as zip_ref:
    zip_ref.extractall()