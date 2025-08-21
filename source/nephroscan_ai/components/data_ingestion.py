import os
import zipfile
import gdown
from nephroscan_ai import logger
from nephroscan_ai.utils.common import get_size
from nephroscan_ai.config.configuration import DataIngestionConfig


import os
import zipfile
import gdown

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        '''
        Fetch data from the url
        '''
        try: 
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = 'https://drive.google.com/uc?/export=download&id='
            
            # ✅ Show progress bar + resume download
            gdown.download(
                prefix + file_id,
                str(zip_download_dir),
                quiet=False,
                resume=True
            )

            logger.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")
            return str(zip_download_dir)

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file into the data directory.
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(str(unzip_path), exist_ok=True)

        # ✅ Convert Path → str
        with zipfile.ZipFile(str(self.config.local_data_file), 'r') as zip_ref:
            zip_ref.extractall(str(unzip_path))
