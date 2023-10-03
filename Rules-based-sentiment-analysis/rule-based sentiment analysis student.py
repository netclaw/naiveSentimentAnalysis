from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string
import re
import collections
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

ps = PorterStemmer()
wl = WordNetLemmatizer()

# new_text = "It was one of the worst movies, 56 - ? despite good ."
# new_text = "the movie was bad. horses, eating!"
new_text = "it is a BAD and HORRIBLE movie!"
# new_text = "this product is nice. i really appreciate these awsome products!"

print("---------------TEXT--------------")
print(new_text)
print("")




print("---------------TOKENIZATION AND LOWER CASE--------------")
## to lower case
# insert_code
new_text=new_text.lower()
## couper la phrase en mots
# insert_code
tokens = word_tokenize(new_text)

print(tokens)
print("---------------NORMALIZATION--------------")
## normalisation
#insert_code
normalized_text1 = [re.sub(r"[^\w\s]", "", word) for word in tokens]
## remove numbers
#insert_code
normalized_text = [re.sub(r"\d+","", word) for word in normalized_text1]
normalized_textf= [ x for x in normalized_text if x!='']
print(normalized_textf)
print("---------------REMOVE STOP WORDS--------------")
## charger les stopwords
#insert_code
stop_words = set(stopwords.words("english"))
## remove stop words 
#insert_code
filtered_text = [word for word in normalized_textf if word not in stop_words]
print(filtered_text)

# print("---------------STEMMING--------------")
# ## Stemming
# #insert_code
# stemmed_text = [ps.stem(word) for word in filtered_text]
# print(stemmed_text)
print("---------------LEMMATIZING--------------")
## Lemmatizing
#insert_code
lemmatized_text = [wl.lemmatize(word) for word in filtered_text]
print(lemmatized_text)
print("-----------------OCCURENCES--------------------------------")
##occurrence
#insert_code
word_count = collections.Counter(lemmatized_text)
print(word_count)
print("---------------POSITIVITY/NEGATIVITY--------------")
positive_words_path = "/Users/youssefberkia/Documents/folder/machine learning with python/AI_NLP/Rules-based-sentiment-analysis/positive-words.txt"
negative_words_path = "/Users/youssefberkia/Documents/folder/machine learning with python/AI_NLP/Rules-based-sentiment-analysis/negative-words.txt"

###Calculating 
#insert_code
with open(positive_words_path, "r") as positive_file:
    positive_words = positive_file.read().splitlines()

with open(negative_words_path, "r") as negative_file:
    negative_words = negative_file.read().splitlines()


positive_count = sum(word_count[word] for word in word_count if word in positive_words)
negative_count = sum(word_count[word] for word in word_count if word in negative_words)
print([word for word in word_count if word in positive_words])
print([word for word in word_count if word in negative_words])
print(positive_count)
print(negative_count)

print("---------------SENTIMENT OF THE TEXT--------------")
###Deciding if it is postive or negative
#insert_code
if positive_count > negative_count:
    print("The text has a positive sentiment.")
elif negative_count > positive_count:
    print("The text has a negative sentiment.")
else:
    print("The text has a neutral sentiment.")

