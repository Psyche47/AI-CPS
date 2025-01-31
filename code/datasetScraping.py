import requests
import pandas as pd
import os

#print(os.getcwd())
def scrapeDataset(github_url,filename,folder_path):
     try:
          response = requests.get(github_url)

          file_path = os.path.join(folder_path, filename).replace("\\", "/")
          print(file_path)
          if response.status_code == 200:
               with open(file_path, "wb") as file:
                    file.write(response.content)
               return f"Dataset successfully scraped"
          else:
               return f"Failed to scrape dataset, Error code: {response.status_code}"
     except Exception as e:
          return f"An error occured {e}"

github_url = "https://raw.githubusercontent.com/Psyche47/AirQualityUCI_Dataset/main/Dataset/AirQualityUCI.csv"
filename = "AirQualityUCI.csv"
folder_path = "../data"

result = scrapeDataset(github_url, filename, folder_path)
print(result)