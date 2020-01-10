from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
from utils import *


# def count_vectorize()

def get_df_for_author(author, period):
    texts_df = pd.DataFrame(columns=["author", "text"])
    i = 0
    type_dirs = {period: ["./" + author]}
    docs = load_docs([period], type_dirs=type_dirs)
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(remove_latin_library_items(preprocess(doc)))
    rows = []
    for i, cleaned_doc in enumerate(cleaned_docs):
        split_docs = split_count(cleaned_doc, 200)  # splits into 50 word chuncks
        for split_doc in split_docs:
            cur_doc = {"author": author, "text": split_doc}
            rows.append(cur_doc)
        # texts_df[i] = [period, cleaned_doc]
        # print(i, cleaned_doc)
    texts_df = pd.DataFrame(rows)
    return texts_df


def get_text_by_authors(authors):
    frames = []
    for author, period in authors:
        cur_df = get_df_for_author(author, period)
        frames.append(cur_df)

    combined_frames = pd.concat(frames)
    return combined_frames


def load_data():
    authors = [("cicero", "republican"), ("caesar", "republican"), ("nepos", "republican"), ("lucretius", "republican"),
               ("livy", "augustan"), ("ovid", "augustan"), ("horace", "augustan"), ("vergil", "augustan"),
               ("hyginus", "augustan")]

    text_df = get_text_by_authors(authors)
    text_df.to_csv("./text_df/text-rep_aug.csv", index=False)
    text_df = shuffle(text_df)
    y = text_df["author"]
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)
    X = text_df["text"]

    # from medium https://towardsdatascience.com/a-machine-learning-approach-to-author-identification-of-horror
    # -novels-from-text-snippets-3f1ef5dba634 80-20 splitting the dataset (80%->Training and 20%->Validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y
                                                        , test_size=0.2, random_state=1234)

    # defining the bag-of-words transformer on the text-processed corpus # i.e., text_process() declared in II is
    # executed...
    bow_transformer = CountVectorizer().fit(X_train)

    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_train = bow_transformer.transform(X_train)  # ONLY TRAINING DATA

    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_test = bow_transformer.transform(X_test)  # TEST DATA

    return text_bow_train, text_bow_test, y_train, y_test


def main():
    text_bow_train, text_bow_test, y_train, y_test = load_data()

    print(y_test)
    counts = np.bincount(y_test)
    ii = np.nonzero(counts)[0]
    print(list(zip(ii, counts[ii])))


    print(y_test[0])
    print(text_bow_test[0])

    model = MultinomialNB()

    model = model.fit(text_bow_train, y_train)

    print("Train Accuracy")
    print(model.score(text_bow_train, y_train))

    print("Eval Accuracy")

    print(model.score(text_bow_test, y_test))



if __name__ == "__main__":
    main()
