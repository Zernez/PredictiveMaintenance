from tools import cleaner, file_writer, raw_loader
import config as cfg
from pathlib import Path
from time import time

class load_and_clean_data:

    def __init__(self):
        pass
    
    loader = raw_loader.RawLoader()
    cleaner = cleaner.Cleaner()
    file_writer = file_writer.FileWriter()
    start_time = time()

    # Load data
    ats = loader.load_assistive_aids(cfg.FILENAMES_OTHER_RAW['ATS'], cfg.RAW_DATA_DIR)
    home_care = loader.load_home_care(list(cfg.FILENAMES_HC_RAW.values()), cfg.RAW_DATA_DIR)
    iso_classes = loader.load_iso_classes(cfg.FILENAMES_REF['iso_classes'], cfg.REFERENCES_DIR)
    
    # Clean data
    home_care = cleaner.clean_home_care(home_care)
    ats = cleaner.clean_ats(ats, iso_classes)

    # Save data
    file_writer.write_pickle(Path.joinpath(cfg.INTERIM_DATA_DIR, 'ats.pkl'), ats)
    file_writer.write_pickle(Path.joinpath(cfg.INTERIM_DATA_DIR, 'home_care.pkl'), home_care)

    passed_time = time() - start_time
    print(f"Load and clean data took {passed_time}")
