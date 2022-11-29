# Standard libraries
from nltk.stem import WordNetLemmatizer
from nltk import download, word_tokenize
from numpy import array as np_array, ndarray
from keras.layers import Dense, Dropout
from keras import Sequential, optimizers
from inquirer import prompt, Text
from json import loads
from os import getcwd, environ
from random import shuffle, choice
from string import punctuation
from typing import Tuple, List

# Disable Tensorflow messages
environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# ML libraries


class ChatBot:

    def __init__(self):
        download("punkt", quiet=True)
        download("wordnet", quiet=True)
        download('omw-1.4', quiet=True)
        self.data = self.load_dataset()
        self.lemmatizer = WordNetLemmatizer()  # This will help us get the root of each word
        self.words, self.labels, self.train_x, self.train_y = self.create_train_data()
        self.model = self.create_model()

    def load_dataset(self) -> dict:
        ''' Load a dataset specified by the user '''
        answer = prompt([Text("file", message="Dataset", default=f"{getcwd()}/dataset.json")])
        data_file = open(answer["file"]).read()
        return loads(data_file)

    def get_data(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        ''' Get the data from the dataset and sort it out into different data structures '''
        words = set()  # List of all words that appear in all patterns
        labels = set()  # Tags
        data_x = []  # List containing the patterns
        data_y = []  # List containing the tags corresponding to each pattern

        for intent in self.data["categories"]:
            for pattern in intent["patterns"]:
                tokens = word_tokenize(pattern)  # Tokenized pattern
                words.update(tokens)  # Add all tokens (words) to the list of words
                data_x.append(pattern)  # Add the pattern to the list of patterns
                data_y.append(intent["tag"])  # Add the tag to the list of tags
            labels.add(intent["tag"])  # Add the tag as a new class

        words_list = sorted([self.lemmatizer.lemmatize(word.lower())
                            for word in filter(lambda word: word not in punctuation, words)])
        labels_list = sorted(list(labels))

        return (words_list, labels_list, data_x, data_y)

    def create_train_data(self) -> Tuple[List[str], List[str], ndarray, ndarray]:
        ''' Create the train data based on the data in the dataset '''
        words, labels, data_x, data_y = self.get_data()

        train_data = []
        out_empty = [0] * len(labels)

        for idx, pattern in enumerate(data_x):
            input = []  # Vector which represent which words appear in the pattern
            for word in words:
                input.append(1) if word in pattern.lower() else input.append(0)

            output = out_empty.copy()
            output[labels.index(data_y[idx])] = 1  # Set to 1 the index which correspond to this pattern's class
            train_data.append([input, output])

        shuffle(train_data)
        train_data_array = np_array(train_data, dtype=object)
        train_x = np_array(list(train_data_array[:, 0]))
        train_y = np_array(list(train_data_array[:, 1]))

        return (words, labels, train_x, train_y)

    def create_model(self) -> Sequential:
        ''' Create the ML model '''
        model = Sequential()
        model.add(Dense(128, input_shape=(len(self.train_x[0]), ), activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(len(self.train_y[0]), activation="softmax"))
        adam = optimizers.Adam(learning_rate=0.01, decay=1e-6)
        model.compile(loss="categorical_crossentropy", optimizer=adam, metrics=["accuracy"])
        model.fit(x=self.train_x, y=self.train_y, epochs=150, verbose=0)
        return model

    def clean_message(self, message: str) -> List[str]:
        ''' Clean the input message from the user '''
        tokens = word_tokenize(message.lower())
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        return tokens

    def bag_of_words(self, message: str) -> ndarray:
        ''' Create a bag of words model from the user message '''
        tokens = self.clean_message(message)
        bow = [0] * len(self.words)
        for token in tokens:
            for idx, word in enumerate(self.words):
                if token == word:
                    bow[idx] = 1
        return np_array(bow)

    def pred_class(self, message: str) -> List[str]:
        ''' Predict the labels (categories) which might correspond to the user message '''
        bow = self.bag_of_words(message)
        results = self.model.predict(np_array([bow]), verbose=0)[0]  # Get the probabilities for each label
        threshold = 0.5
        y_pred = [[idx, res] for idx, res in enumerate(results) if res > threshold]
        y_pred.sort(key=lambda x: x[1], reverse=True)  # Sorting by values of probability in decreasing order
        return [self.labels[r[0]] for r in y_pred]  # Return the labels for each of the results

    def get_response(self, results: List[str]) -> str:
        ''' Get a response based on the highes probable result label '''
        if len(results) > 0:
            tag = results[0]  # Get the label of the result with the highest probability
            for intent in self.data["categories"]:
                if intent["tag"] == tag:
                    return choice(intent["responses"])

        return "Sorry, I didn't understand you :("

    def chat(self):
        ''' Chat with the bot '''
        print("Press '0', 'q', 'end' or 'bye' if you don't want to chat anymore")
        while True:
            message = input("user: ")
            if message in ['0', 'q', 'end', 'bye']:
                print("bot: Good bye!")
                break
            results = self.pred_class(message)
            response = self.get_response(results)
            print(f"bot: {response}")


def main():
    chatbot = ChatBot()
    chatbot.chat()


if __name__ == "__main__":
    main()
