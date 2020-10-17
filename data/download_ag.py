from google_drive_downloader import GoogleDriveDownloader as gdd

data = {
    "train": "1XlAh2j9ngbb31SWA5-gyF293JG7dUqMO",
    "test": "1LpXt1p8_u25UDZnLhO7XfEAbuXo2x4fY",
}

if __name__ == "__main__":
    for filename, file_id in data.items():
        gdd.download_file_from_google_drive(
            file_id=file_id, dest_path=f"./{filename}.csv.zip", unzip=False
        )
