from pathlib import Path
import pandas as pd
from zipfile import ZipFile

def dataset_unzipping():
  data_path = Path(__file__).parent.parent / "data"
  zip_files_in_raw_dir = (data_path / "raw").glob("*.zip")

  for zip_file in zip_files_in_raw_dir:
    try:
      with ZipFile(zip_file, "r") as z:
        z.extractall((data_path / "raw" / zip_file.stem))
      zip_file.unlink()
    except Exception as e:
      print(f"Unzipping {zip_file} failed: {e}")

def dataset_transferring():
  #TODO
  return


if __name__ == "__main__":
  dataset_unzipping()
