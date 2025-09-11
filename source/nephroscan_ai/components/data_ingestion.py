import os
import zipfile
import gdown
import shutil
from nephroscan_ai import logger
from nephroscan_ai.config.configuration import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self) -> str:
        """
        Fetch data from the URL (Google Drive).
        """
        try:
            dataset_url = self.config.source_URL
            zip_download_dir = self.config.local_data_file

            os.makedirs(os.path.dirname(zip_download_dir), exist_ok=True)
            logger.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            file_id = dataset_url.split("/")[-2]
            prefix = "https://drive.google.com/uc?/export=download&id="

            gdown.download(
                prefix + file_id,
                str(zip_download_dir),
                quiet=False,
                resume=True
            )

            logger.info(f"✅ Download complete: {zip_download_dir}")
            return str(zip_download_dir)

        except Exception as e:
            raise e

    def extract_zip_file(self):
        """
        Extracts the zip file and fixes the double nesting issue.
        Final structure: artifacts/data_ingestion/CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone/{Cyst, Normal, Stone, Tumor}
        """
        unzip_path = self.config.unzip_dir
        os.makedirs(str(unzip_path), exist_ok=True)

        with zipfile.ZipFile(str(self.config.local_data_file), "r") as zip_ref:
            zip_ref.extractall(str(unzip_path))

        # ✅ Fix nested folder problem
        base_folder = os.path.join(unzip_path, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")
        nested_folder = os.path.join(base_folder, "CT-KIDNEY-DATASET-Normal-Cyst-Tumor-Stone")

        if os.path.exists(nested_folder):
            logger.info("⚠️ Nested folder detected. Flattening structure...")
            for item in os.listdir(nested_folder):
                src = os.path.join(nested_folder, item)
                dst = os.path.join(base_folder, item)
                shutil.move(src, dst)

            shutil.rmtree(nested_folder)  # remove the extra nested folder
            logger.info("✅ Dataset structure fixed. Now only 4 class folders remain.")
