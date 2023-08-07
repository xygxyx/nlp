import os,requests,tarfile,datetime,torch
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
import pandas as pd

model = SentenceTransformer('all-MiniLM-L6-v2');

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
def convert_field(name,text,vals):
    val = text 

    if name == 'rating':
        val = int(float(text))

    # multiply rating if it was deemed helpfull substract that rating that many times otherwise
    if name == 'helpful' and 'rating' in vals:
        a,b=text.split('of'); a=int(a); b=int(b)
        ratio=a/b; agreed=int(b*ratio); disagreed=b-agreed
        rating=int(vals['rating'])
        vals['rating']+=agreed*rating
        vals['rating']-=disagreed*rating
        

    # convert date strings in '11/14/2006' format to simple rising integer so nn can observe and learn date related moods
    if name == 'date':
        val = datetime.strptime(text, '%m/%d/%Y').timestamp()

    # merge spaces remove lines.
    if name in ['title','review_text']:
        val = ' '.join(text.split()).strip()

        # calc fixed size vector of in this case 384 numbers = embedding for given text
        val = model.encode(val)

    return val

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
                        field_values[field.name] = convert_field(field.name,field.get_text().strip(),field_values)
                        
                    # Append the dictionary to the data list
                    data.append(field_values)

    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)

    # Randomize the rows to prevent nonuniform collection issues
    df = df.sample(frac=1).reset_index(drop=True)

    # Create scalers
    mm = MinMaxScaler(); le = LabelEncoder()

    # Fit the 'date' column from 0 to 1 so net can track and learn potential seasonal moods
    df['date'] = mm.fit_transform(df[['date']])

    # Fit the 'sentiment_class' column to numerical values from 0 to 1
    df['sentiment_class'] = le.fit_transform(df['sentiment_class'])
    df['sentiment_class'] = mm.fit_transform(df[['sentiment_class']])

    df.to_csv(dataset+'.csv', index=False)
    return df

dir = 'sorted_data_acl'

# Download and unpack Dataset in its custom concatenated XML's in dir structures 
if not os.path.isdir(dir):
    download_dataset('https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz')

# Load the CSV file into a DataFrame if it exists, or create it from XML's in dirs for first time
df = pd.read_csv(dir+'.csv') if os.path.exists(dir+'.csv') else xml_to_csv(dir)
    
# Print the DataFrame
print(df)
