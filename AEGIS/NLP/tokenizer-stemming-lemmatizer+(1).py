
# coding: utf-8

# ### In this tutorial, we will look at some of the tokenizers available in nltk

# In[3]:


## Tokenization using NLTK
# word_tokenize
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
s = "Good muffins cost $3.80 in New York.\nDr. Ram Please buy me two of them.\nThanks."
print("Sentence: \n\n"+s) 
print("\nword_tokenize output")
print(word_tokenize(s))

print("\nsplit tokenize output")
print(s.split())
print("\n")


# In[4]:


# word_tokenize
import nltk
from nltk.tokenize import wordpunct_tokenize
s = "Good muffins cost $3.80 in New York.\nDr. Ram Please buy me two of them.\nThanks."
print("Sentence: \n\n"+s) 
print("\nwordpunct_tokenize output")
print(wordpunct_tokenize(s))
print("\n")


# In[5]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
s2= "Good muffins cost $3.80 in New York. Dr. Ram Please buy me two of them.Thanks."
print("Sentence: \n\n"+s2) 
print("\nsent_tokenize output")
print(sent_tokenize(s2))
print("\nword_tokenize output")
for t in sent_tokenize(s2):
    print(word_tokenize(t))
print("\n")


# In[6]:


# LineTokenizer
import nltk
from nltk.tokenize import LineTokenizer


# LineTokenizer can be used to split strings containing newline characters

# In[7]:


s = "I love kites.\nI like cricket.\nI like football.\n"

print("Sentences: ") 
print(s)
print("LineTokenizer...")
print(LineTokenizer().tokenize(s))
print("\nword_tokenizer... ")
for sent in LineTokenizer().tokenize(s):
    print(word_tokenize(sent))


# In[8]:


from nltk.tokenize import RegexpTokenizer


# RegexpTokenizer allows us to provide regular expressions as delimiters
# The material between the tokens is discarded. 

# In[9]:


s = "Petrol price has gone upto Rs.75.89 on 01/02/2017. John and Mrs. Thomas are thinking of using electric scooters."
tokenizer = RegexpTokenizer('Rs\.[\d]+\.[\d]+')
print("Sentence: "+s)
print("\nRegexpTokenizer...")
print(tokenizer.tokenize(s))
print("\n")


# In[10]:


#Let us say we want to extract all words beginning with an uppercase character
tokenizer = RegexpTokenizer('[A-Z]\w*\S+')
print(tokenizer.tokenize(s))


# #### SExprTokenizer : Tokenizes parenthesized expressions in a string 

# In[11]:


from nltk.tokenize import SExprTokenizer


# In[12]:


s = '?(a(b c)d)ef(g(h(i)))'
print("Sentence: "+s)
print("\nSExprTokenizer...")
print(SExprTokenizer().tokenize(s))
print("\n")


# #### TreebankWordTokenizer is standard tokenizer tool used and does a decent job

# In[13]:


#TreebankWordTokenizer
from nltk.tokenize import TreebankWordTokenizer


# In[14]:


s = "Good muffins cost $3.80 in New York. Dr. Ram Please buy me two of them. Thanks."
print("Sentence: "+s)
print("\nTreebankWordTokenizer...")
print(TreebankWordTokenizer().tokenize(s))
print("\n")


# In[15]:


s= "@Nikes: This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
print("\nSentence: "+s)
print(TreebankWordTokenizer().tokenize(s))


# #### The previous tokenizers fail badly for tweets, TweetTokenizer can be used to tokenize tweets

# In[16]:


from nltk.tokenize import TweetTokenizer


# In[17]:


tknzr = TweetTokenizer()
s0 = "@Nike: This is a cooool #dummysmiley: :-) :-P <3 and some arrows < > -> <--"
tknzr.tokenize(s0)


# In[18]:


#**WordNet Lemmatizer**
#Lemmatize using WordNet’s built-in morphy function. Returns the input word unchanged if it cannot be found in WordNet.
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
print(wnl.lemmatize('computed'))
print(wnl.lemmatize('computed','v'))
print(wnl.lemmatize('nationality'))


# In[19]:


#**SnowballStemmer**
#For Snowball Stemmer, which is based on Snowball Stemming Algorithm, can be used in NLTK like this:
from nltk.stem import SnowballStemmer
print(" ".join(SnowballStemmer.languages))


# In[20]:


snowball_stemmer = SnowballStemmer('english')
#snowball_stemmer.stem('maximum')
#snowball_stemmer.stem('presumably')
print(snowball_stemmer.stem('computing'))
print(snowball_stemmer.stem('nationality'))


# In[21]:


from nltk.stem.snowball import GermanStemmer
stemmer = GermanStemmer()
stemmer.stem("Autobahnen")


# In[22]:


#for more details and examples see http://www.nltk.org/api/nltk.tokenize.html1


# In[33]:


from nltk.corpus import stopwords
nltk.download("stopwords")
text="Sachin Ramesh Tendulkar (/ˌsətʃɪn tɛnˈduːlkər/ (About this sound listen); born 24 April 1973) is a former Indian cricketer and a former captain, regarded as one of the greatest batsmen of all time.[4] The highest run scorer of all time in International cricket, Tendulkar took up cricket at the age of eleven, made his Test debut on 15 November 1989 against Pakistan in Karachi at the age of sixteen, and went on to represent Mumbai domestically and India internationally for close to twenty-four years. He is the only player to have scored one hundred international centuries, the first batsman to score a double century in a One Day International, the holder of the record for the most number of runs in both ODI and Test cricket, and the only player to complete more than 30,000 runs in international cricket.[5]"
stopwordsList=set(stopwords.words('english'))
stopwordsListUpdated=list(stopwordsList)
punctuationList=["{", "}", "]", "[", ",", ";", ".", "/"]
stopwordsListUpdated+= punctuationList            
textlower=text.lower()
textlower
tokens=word_tokenize(textlower)
token_updated=[]
for token in tokens:
    if token not in stopwordsListUpdated and not token.isdigit():
        lemma=snowball_stemmer.stem(token)
        token_updated.append(lemma)
print("TOken: ",len(token_updated))
print("Unique words :",len(set(token_updated)))

