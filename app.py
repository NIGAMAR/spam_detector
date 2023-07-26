import pandas as pd
import re
import nltk as nlp
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

df = pd.read_csv('data/spam_collection.csv',
                 sep='\t', names=['label', 'message'])

nlp.download('stopwords')
nlp.download('punkt')
nlp.download('wordnet')

stemmer = PorterStemmer()  # has accuracy of approx 98.20%
lematizer = WordNetLemmatizer()  # has accuracy of approx 98.47%
corpus = []

# preprocessing the data
for i in range(0, len(df)):
    # replace everything except words from the msg
    review = re.sub('[^a-zA-Z]', ' ', df['message'][i])
    review = review.lower()
    review = nlp.word_tokenize(review)
    review = [lematizer.lemmatize(word)
              for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

# using bag of words to vectorize the sentences
cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus)
y = pd.get_dummies(df['label'])
y = y.iloc[:, 1].values

# perform train test splitting

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# training model using Naive bayes classifier

spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
