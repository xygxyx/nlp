import pandas as pd
from bs4 import BeautifulSoup
import os,requests,tarfile

# Download and unpack
def download_dataset(url):
    filename = os.path.basename(url)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.raw.read())
    tar = tarfile.open(filename, "r:gz")
    tar.extractall()
    tar.close()
    os.remove(filename)

# Convert Dataset from Dirs with XML list files to easy and fast to load CSV
def xml_to_csv(dataset):
    # Initialize an empty list to store rows
    data = []

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(dataset):
        for filename in filenames:
            # Check if the file is a .review file
            if filename.endswith('.review'):
                # Extract category from the directory name
                category = os.path.basename(dirpath)
                # Extract sentiment class from the file name
                sentiment_class = filename.split('.')[0]
                
                # Open and parse the HTML file
                with open(os.path.join(dirpath, filename), 'r') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    
                # Find all review tags
                review_tags = soup.find_all('review')
                
                for review_tag in review_tags:
                    # Initialize a dictionary to store subtag values
                    subtag_values = {'category': category, 'sentiment_class': sentiment_class}
                    
                    # Find all custom, non-HTML tags within the review_tag
                    subtags = review_tag.find_all(recursive=False)
                    
                    for subtag in subtags:
                        # Store subtag name and value. cleanup htm <br> etc. and normalize whitespace. 
                        subtag_values[subtag.name] = ' '.join(subtag.get_text().split()).strip()
                        
                    # Append the dictionary to the data list
                    data.append(subtag_values)

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='ignore')

    df.to_csv(dataset+'.csv', index=False)
    return df

# Download and unpack Dataset in its custom concatenated XML's in dir structures 
download_dataset('https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz')

dir = 'sorted_data_acl'

# Load the CSV file into a DataFrame, Create it from XML's in dirs for first time if needed
df=xml_to_csv(dir) if not os.path.exists(dir+'.csv') else pd.read_csv(dir+'.csv')
    
# Print the DataFrame
print(df)
