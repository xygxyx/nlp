from bs4 import BeautifulSoup

# Create an empty list to store the objects
objects = []

# Open the XML file
fname='books/negative.review'

# Open the HTML file
with open(fname, 'r') as file:
    # Parse the HTML file using BeautifulSoup
    soup = BeautifulSoup(file, 'html.parser')

# Find the title element of the HTML page
ob=[]
for r in soup.findAll('review'):
    o={}
    # Print the text of the title element
    for ch in r.find_all(recursive=False):
    # Get the tag name and content of the child element
        o[ch.name]=ch.text.strip()
    ob.append(o)

a=0