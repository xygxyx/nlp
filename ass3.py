from bs4 import BeautifulSoup
import os,requests,tarfile
import pandas as pd

# Download and unpack XML tar.gz Dataset
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

# Here we cleanup text to prepare it for load to neural network
def cleanup_text(text):
    
    # merge spaces remove lines.
    return ' '.join(text.split()).strip()

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
                    field_values = {'category': category, 'sentiment_class': sentiment_class}
                    
                    # Find all fields stored in flat review tag hierarchy
                    fields = review_tag.find_all(recursive=False)
                    
                    for field in fields:
                        # Store subtag name and value. cleanup htm <br> etc. and normalize whitespace. 
                        field_values[field.name] = cleanup_text(field.get_text())
                        
                    # Append the dictionary to the data list
                    data.append(field_values)

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # Convert columns to numeric where possible
    df = df.apply(pd.to_numeric, errors='ignore')

    df.to_csv(dataset+'.csv', index=False)
    return df


# Download and unpack Dataset in its custom concatenated XML's in dir structures 
download_dataset('https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz')

dir = 'sorted_data_acl'

# Load the CSV file into a DataFrame if it exists, or create it from XML's in dirs for first time
df = pd.read_csv(dir+'.csv') if os.path.exists(dir+'.csv') else xml_to_csv(dir)
    
# Print the DataFrame
print(df)
