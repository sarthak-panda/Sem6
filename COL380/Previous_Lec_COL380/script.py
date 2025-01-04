import os
import requests
from bs4 import BeautifulSoup

# URL of the website containing the PDF links
base_url = "https://www.cse.iitd.ac.in/~subodh/courses/COL380/pdfslides/"

# Create a directory to save the PDFs
save_dir = "pdf_downloads"
os.makedirs(save_dir, exist_ok=True)

# Fetch the webpage content
response = requests.get(base_url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, "html.parser")
    
    # Find all <a> tags with href attributes ending in .pdf
    pdf_links = soup.find_all("a", href=lambda href: href and href.endswith(".pdf"))
    
    for link in pdf_links:
        pdf_url = base_url + link["href"]
        pdf_name = link["href"].split("/")[-1]
        save_path = os.path.join(save_dir, pdf_name)
        
        # Download and save the PDF
        print(f"Downloading {pdf_name}...")
        pdf_response = requests.get(pdf_url)
        with open(save_path, "wb") as pdf_file:
            pdf_file.write(pdf_response.content)
        print(f"Saved {pdf_name} to {save_dir}")
else:
    print(f"Failed to access {base_url}. Status code: {response.status_code}")
