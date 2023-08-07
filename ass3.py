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
    df['date'] = mm.fit_transform(df['date'])

    # Convert and Fit the 'sentiment_class' column to numerical values from 0 to 1
    df['sentiment_class'] = le.fit_transform(df['sentiment_class'])
    df['sentiment_class'] = mm.fit_transform(df['sentiment_class'])

    # Convert and Fit the 'category' column to numerical values from 0 to 1
    df['category'] = le.fit_transform(df['category'])
    df['category'] = mm.fit_transform(df['category'])

    # Fit the 'rating' column to numerical values from 0 to 1
    df['rating'] = mm.fit_transform(df['rating'])

    df.to_csv(dataset+'.csv', index=False)
    return df

dir = 'sorted_data_acl'

# Download and unpack Dataset in its custom concatenated XML's in dir structures 
if not os.path.isdir(dir):
    download_dataset('https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz')

# Load the CSV file into a DataFrame if it exists, or create it from XML's in dirs for first time
df = pd.read_csv(dir+'.csv') if os.path.exists(dir+'.csv') else xml_to_csv(dir)


# Keep only for the training important columns
df = df.filter('category','date','rating','title','review_text','sentiment_class')
print(df)

# expand embedding vectors to additional new 384 input neurons
for i in range(len(df['title'].iloc[0])):
    df[f'title_{i}'] = df['title'].apply(lambda x: x[i])

# expand embedding vectors to additional new 384 input neurons
for i in range(len(df['review_text'].iloc[0])):
    df[f'review_text_{i}'] = df['review_text'].apply(lambda x: x[i])

# Remove embedding storages
df = df.drop(['title', 'review_text'], axis=1)

# Display the resulting DataFrame
print(df)

# Move the 'sentiment_class' column to the last position so it can be excluded from training
df.insert(len(df.columns), 'sentiment_class', df.pop('sentiment_class'))

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
train_data = ReviewDataset(train)
test_data = ReviewDataset(test)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

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