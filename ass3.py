import os,requests,tarfile,torch
from dateutil import parser
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

# Convert Dataset from Dirs with XML list files to easy and fast to load but raw text CSV
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
                sentiment = filename.split('.')[0]
                
                # Open and parse list of concatenated XML files that only HTML parser can handle
                with open(os.path.join(dirpath, filename), 'r') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    
                # Find all review tags
                review_tags = soup.find_all('review')
                
                for review_tag in review_tags:

                    # Extract field values from dir and file names
                    field_values = {'category': category, 'sentiment': sentiment}
                    
                    # Find all fields stored in flat <review> tag hierarchy
                    fields = review_tag.find_all(recursive=False)

                    # Move helpful as last so it can influence now already processed ratings field
                    helpful_index = next((i for i, field in enumerate(fields) if field.name == 'helpful'), None)
                    helpful_field = fields.pop(helpful_index); fields.append(helpful_field)
                    
                    # Extract field name and value.
                    for field in fields:
                        
                        # Convert htm <br> etc to text. and normalize whitespace. 
                        field_values[field.name] = ' '.join(field.get_text().split()).strip()

                    # Append the dictionary to the data list
                    data.append(field_values)
    
    # Convert the list of dictionaries into a pandas DataFrame
    df = pd.DataFrame(data)
    df.to_csv(dataset+'.csv', index=False)
    return df

# Here we convert all fields to numbers to prepare it for load to neural network
def prepare_field(name,text,df):
    val = text 

    if name == 'rating':
        val = int(float(text))

    # multiply rating. if it was deemed helpfull substract that rating that many times otherwise
    if name == 'helpful' and 'rating' in df.columns and 'of' in text:
        a,b=text.split('of'); a=int(a); b=int(b)
        ratio=a/b; agreed=int(b*ratio); disagreed=b-agreed
        rating=int(df['rating'])

        # Say 1000 people agreed and 1000 disagreed and rating was 2 stars
        # so we add 2000 2 star ratings and substract as well thus having no impact  
        df['rating']+=agreed*rating
        df['rating']-=disagreed*rating
        
    # convert date strings in '14 nov, 2006' format to simple rising integer as seconds since epoch
    # so nn can observe and learn date weather etc related seasonal moods from data
    if name == 'date':
        try:
            val = parser.parse(text).timestamp()
        except:
            pass

    # merge spaces remove lines.
    if name in ['title','review_text']:
        val = ' '.join(text.split()).strip()

        # calc fixed size vector of in this case 384 numbers = embedding for given text
        #val = model.encode(val)

    return val

# Here we reduce and convert all dataset fields to numbers to prepare it for load to neural network
def prepare_dataset(df):

    # Keep only for the training important columns
    df = df.filter('category','date','rating','title','review_text','sentiment')

    # Drop incomplete rows 
    df = df.dropna(how='any')

    # Drop rows where the 'sentiment' column is equal to 'unlabeled'
    df = df[df['sentiment'] != 'unlabeled']

    # Infer the dtypes of the object columns
    df = df.infer_objects()
    df = pd.DataFrame({column: pd.to_numeric(df[column], errors='ignore') for column in df})

    # convert all fields differently some are numbers. convert texts to embeddings
    for column in df:
        df[column] = prepare_field(column,df[column],df)

    # Randomize the rows to prevent nonuniformities introduced during collection to cause issues
    # df = df.sample(frac=1).reset_index(drop=True)

    # Create scalers
    mm = MinMaxScaler(); le = LabelEncoder()

    # Fit the 'date' column from 0 to 1 so net can track and learn potential seasonal moods
    df['date'] = mm.fit_transform(df['date'])

    # Convert and Fit the 'sentiment' column to numerical values from 0 to 1
    df['sentiment'] = le.fit_transform(df['sentiment'])
    df['sentiment'] = mm.fit_transform(df['sentiment'])

    # Convert and Fit the 'category' column to numerical values from 0 to 1
    df['category'] = le.fit_transform(df['category'])
    df['category'] = mm.fit_transform(df['category'])

    # Fit the 'rating' column to numerical values from 0 to 1
    df['rating'] = mm.fit_transform(df['rating'])

    # expand embedding vectors from 2 tex fields to 2x additional new 384 input neurons
    for field in ['title','review_text']
        for i in range(len(df[field].iloc[0])):
            df[f'{field}_{i}'] = df[field].apply(lambda x: x[i])

        # Remove no longer needed embedding storage arrays not passable to nn
        df = df.drop([field], axis=1)

    # Move the 'sentiment' column to the last position so it can now be properly excluded from training
    df.insert(len(df.columns), 'sentiment', df.pop('sentiment'))

    return df



dir = 'sorted_data_acl'

# Download and unpack Dataset in its custom concatenated XML's in dir structures 
if not os.path.isdir(dir):
    download_dataset('https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz')

# Load the CSV file into a DataFrame if it exists, or create it from XML's in dirs for first time
df = pd.read_csv(dir+'.csv') if os.path.exists(dir+'.csv') else xml_to_csv(dir)

# convert to numbers
df = prepare_dataset(df)

# Split the data into training and test sets
train, test = train_test_split(df, test_size=0.2)

# Define a custom dataset
class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float), torch.tensor(self.data.iloc[idx, -1], dtype=torch.long)

# Create data loaders
train_data = ReviewDataset(train); train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_data  = ReviewDataset(test);  test_loader  = DataLoader(test_data,  batch_size=32, shuffle=True)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__(); ins=len(df.columns)
        self.fc1 = nn.Linear(ins - 1   , int(ins/4))
        self.fc2 = nn.Linear(int(ins/4), int(ins/8))
        self.fc2 = nn.Linear(int(ins/8), int(ins/16))
        self.fc3 = nn.Linear(int(ins/16), 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize the network, the optimizer and the loss function
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Train the network
for epoch in range(100):  # number of epochs
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Test the network
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))