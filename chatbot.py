import json
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity 
# Download NLTK data (free)
nltk.download('punkt')
nltk.download('wordnet')

class FreeChatbot:
    def __init__(self, intents_file="intents.json"):
        self.lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenize, stop_words='english')
        self.load_intents(intents_file)
        
    
    def _tokenize(self, text):
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        return [self.lemmatizer.lemmatize(token) for token in tokens]
        
    def load_intents(self, file_path):
        with open(file_path) as file:
            self.intents = json.load(file)
        
        # Prepare training data
        self.patterns = []
        self.responses = []
        
        for intent in self.intents['intents']:
            for pattern in intent['patterns']:
                self.patterns.append(pattern)
                self.responses.append(intent['responses'])
        
        # Vectorize patterns
        self.X = self.vectorizer.fit_transform(self.patterns)
    
    def get_response(self, user_input):
        # Vectorize input
        input_vec = self.vectorizer.transform([user_input])
        
        # Calculate similarity
        similarities = cosine_similarity(input_vec, self.X)
        best_match_idx = np.argmax(similarities)
        
        # Get random response from matched intent
        return random.choice(self.responses[best_match_idx])

if __name__ == "__main__":
    print("Initializing free chatbot...")
    bot = FreeChatbot()
    
    print("Chatbot ready! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
            
        response = bot.get_response(user_input)
        print("Bot:", response)