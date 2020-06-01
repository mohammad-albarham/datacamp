# Exercise_1 
# Tokenize the article into sentences: sentences
sentences = nltk.sent_tokenize(article)

# Tokenize each sentence into words: token_sentences
token_sentences = [nltk.word_tokenize(sent) for sent in sentences]

# Tag each tokenized sentence into parts of speech: pos_sentences
pos_sentences = [nltk.pos_tag(sent) for sent in token_sentences] 

# Create the named entity chunks: chunked_sentences
chunked_sentences = nltk.ne_chunk_sents(pos_sentences, binary=True)

# Test for stems of the tree with 'NE' tags
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, "label") and chunk.label() == "NE":
            print(chunk)


--------------------------------------------------
# Exercise_2 
#1
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)
#2
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

#3
# Create the defaultdict: ner_categories
ner_categories = defaultdict(int)

# Create the nested for loop
for sent in chunked_sentences:
    for chunk in sent:
        if hasattr(chunk, 'label'):
            ner_categories[chunk.label()] += 1
            
# Create a list from the dictionary keys for the chart labels: labels
labels = list(ner_categories.keys())

# Create a list of the values: values
values = [ner_categories.get(v) for v in labels]

# Create the pie chart
plt.pie(values, labels=labels, autopct='%1.1f%%', startangle=140)

# Display the chart
plt.show()


--------------------------------------------------
# Exercise_3 
# Import spacy
import spacy 

# Instantiate the English model: nlp
nlp = spacy.load('en', tagger=False, parser=False, matcher=False)
# Create a new document: doc
doc = nlp(article)

# Print all of the found entities and their labels
for ents in doc.ents:
    print(ents.label_ , ents.text)

--------------------------------------------------
# Exercise_4 
# Create a new text object using Polyglot's Text class: txt
txt = Text(article)

# Print each of the entities found
for ent in txt.entities:
    print(ent)
    
# Print the type of ent
print(type(ent))


--------------------------------------------------
# Exercise_5 
# Create the list of tuples: entities
entities = [(ent.tag, ' '.join(ent)) for ent in txt.entities]

# Print entities
print(entities)

--------------------------------------------------
# Exercise_6 
# Initialize the count variable: count
count = 0

for entity in txt.entities:
    # Check whether the entity contains 'Márquez' or 'Gabo'
    if "Márquez" in entity or "Gabo" in entity:
        # Increment count
        count += 1


# Print count
print(count)

# Calculate the percentage of entities that refer to "Gabo": percentage
percentage = count / len(txt.entities)
print(percentage)


--------------------------------------------------
