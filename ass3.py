import os,requests,tarfile,torch,sys,pickle
from dateutil import parser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
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

# Read Dataset from Dirs with XML list files
def read_xml_dataset(dataset):
    # Initialize an empty list to store rows
    data = []; k=0

    # Walk through the directory
    for dirpath, dirnames, filenames in os.walk(dataset):
        for filename in filenames:

            # Check if the file is a .review file
            if filename.endswith('.review'):

                # Extract category from the directory name
                category = os.path.basename(dirpath)

                # Extract sentiment class from the file name
                sentiment = filename.split('.')[0]

                # we skip bulk 72k of unlabeled data since embeding 
                # saved in this csv for them is slow and not needed for training
                if sentiment not in ['positive','negative']:
                    continue
                
                # Open and parse list of concatenated XML files that only HTML parser can handle
                with open(os.path.join(dirpath, filename), 'r') as file:
                    soup = BeautifulSoup(file, 'html.parser')
                    
                # Find all review tags
                review_tags = soup.find_all('review')

                for review_tag in review_tags:
                    print(k,'embedding',dirpath,filename,'            ',end='\r'); k+=1

                    # Extract field values from dir and file names
                    field_values = {'category': category, 'sentiment': sentiment}
                    
                    # Find all fields stored in flat <review> tag hierarchy
                    fields = review_tag.find_all(recursive=False)

                    # Extract field name and value.
                    for field in fields:
                        field_values[field.name] = field.get_text().strip()

                    # Append the dictionary to the data list
                    data.append(field_values)
    
    # Convert the list of dictionaries into a pandas DataFrame 
    df = pd.DataFrame(data)
    return df

# Here we reduce and convert all dataset fields to numbers to prepare it for load to neural network
def prepare_dataset(df):

    # Keep only for the training important columns
    df = df.filter(items=['index','category','date','rating','helpful','title_emb','review_text_emb','sentiment'])
    
    # Rename emb columns
    df = df.rename(columns={'title_emb':'title','review_text_emb':'review'})

    # missing review text is fine as is title or helpfull. they are all just additional info
    fields = ['helpful','category']
    df[fields] = df[fields].fillna('')

    # Drop rows but only where important rating field is missing missing rest is ok
    # i.e if it has rating and angry title but not text etc
    df = df.dropna(subset=['rating','date'],how='any')

    # Reset the index of the DataFrame after deleting rows. So we can iterate linearly again
    df = df.reset_index(drop=True)

    # process / convert all fields differently some are in need of complex funcs some are dates in strings.
    for column in df:
        for i in range(len(df)):
            val = df.loc[i,column] 

            # multiply rating. if it was deemed helpfull substract that rating that many times otherwise
            if column == 'helpful' and 'of' in str(val):
                a,b=str(val).split('of'); a=int(a); b=int(b)
                ratio=a/b; agreed=int(b*ratio); disagreed=b-agreed
                rating=int(float(df.loc[i,'rating']))

                # Say 1000 people agreed and 1000 disagreed and rating was 2 stars
                # so we add 2000 2 star ratings and substract as well thus having no impact  
                df.loc[i,'rating']+=agreed*rating
                df.loc[i,'rating']-=disagreed*rating
                
            # convert date strings in '14 nov, 2006' format to simple rising integer as seconds since epoch
            # so nn can observe and learn date weather etc related seasonal moods from data
            if column == 'date':
                try:
                    val = parser.parse(str(val)).timestamp()
                except:
                    pass        
            
            df.loc[i,column] = val

    # Drop the no longer needed string column 'helpful' from the DataFrame
    df = df.drop('helpful', axis=1)
        

    # Here we would normally randomize the rows order once to prevent nonuniformities introduced during collection to cause issues
    # but pythorch does this every epoch via shuffle param which is way more efficient
    # df = df.sample(frac=1).reset_index(drop=True)
 
    # Create scalers
    mm = MinMaxScaler(); le = LabelEncoder()

    # Fit the 'date' column from 0 to 1 so net can track and learn potential seasonal moods
    df['date'] = mm.fit_transform(df['date'].values.reshape(-1, 1))

    # Convert and Fit the 'sentiment' column to numerical values from 0 to 1
    df['sentiment'] = le.fit_transform(df['sentiment'])
    df['sentiment'] = mm.fit_transform(df['sentiment'].values.reshape(-1, 1))

    # Convert and Fit the 'category' column to numerical values from 0 to 1
    df['category'] = le.fit_transform(df['category'])
    df['category'] = mm.fit_transform(df['category'].values.reshape(-1, 1))

    # Fit the 'rating' column to numerical values from 0 to 1
    df['rating'] = mm.fit_transform(df['rating'].values.reshape(-1, 1))
    return df



dir = 'sorted_data_acl'

# If needed download and unpack Dataset in its custom concatenated XML's in dir structures 
if not os.path.isdir(dir):
    download_dataset('https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz')

# Load the CSV file with raw texts into a DataFrame if it exists, 
if os.path.isfile(dir+'.csv'):
    df = pd.read_csv(dir+'.csv') 
else:
    # Or create it from XML's in dirs and cache it in processed csv for faster next load of raw texts
    df = read_xml_dataset(dir)
    df.to_csv(dir+'.csv', index=False)

# Convert to 0 - 1 normalized numbers processible by nn
df = prepare_dataset(df)

# Load two 374 floats embeding vectors precalculated from two input text fields title and review text
for field in ['title','review_text']:
    if os.path.isfile(field+'.emb'):
        with open(field+'.emb','rb') as file:
            ed=pickle.load(file)
            ed.reset_index(drop=True, inplace=True)
            df.reset_index(drop=True, inplace=True)
            newcols=list(df.columns)+list(ed.columns)

            # Append 384 new embeding cols to our DataFrame
            df = pd.concat([df, ed],axis=1,ignore_index=True,)
            df.columns = newcols
    else:
        # Or calculate and cache them
        e = model.encode(df[field])
        ed = pd.DataFrame(e)
        ed.columns = [f'{field[0]}{i}' for i in range(len(e[0]))]
        with open(field+'.emb','wb') as file:
            pickle.dump(ed,file)

# Move the 'sentiment' column to the last position so it can now be properly excluded from training
o=df.pop('sentiment')
df.insert(len(df.columns), 'sentiment',o )

# Split the data into training and test sets
train, test = train_test_split(df, test_size=0.2)

# Define a custom dataset
class ReviewDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float), torch.tensor(self.data.iloc[idx, -1], dtype=torch.float)

# Create data loaders
train_data = ReviewDataset(train); 
test_data  = ReviewDataset(test);  

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=32, shuffle=True)

# Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__(); ins=len(df.columns)-1
        self.fc1 = nn.Linear(ins, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x
    
# Initialize the network, the optimizer and the loss function
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()
print_every = 1
num_epochs = 10
batch = 0

# Train the network
for epoch in range(num_epochs):  # number of epochs
    for inputs, labels in train_loader:
        batch+=1
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels)
        loss.backward()
        optimizer.step()
        if batch % print_every == 0:  # print_every can be, e.g., 10 to print every 10 batches
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch}/{len(train_loader)}], Loss: {loss.item():.4f}",'                                                 ',end='\r')

# Test the network
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        outputs = model(inputs)
        outputs = torch.sigmoid(outputs)
        predicted = (outputs > 0.5).float().view(-1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (100 * correct / total))