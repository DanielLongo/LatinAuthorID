from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC, SVC
from sklearn.utils import shuffle
import _pickle as cPickle
import numpy as np
import pandas as pd
from utils import *

# def count_vectorize()

SPLIT_LENGTH = 200


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
        split_docs = split_count(cleaned_doc, SPLIT_LENGTH)  # splits into 50 word chuncks
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
    # authors = [("cicero", "republican"), ("caesar", "republican"), ("nepos", "republican"), ("lucretius", "republican"),
    #            ("livy", "augustan"), ("ovid", "augustan"), ("horace", "augustan"), ("vergil", "augustan"),
    #            ("hyginus", "augustan")]

    republican = [("cicero", "republican"), ("caesar", "republican"), ("nepos", "republican"),
                  ("lucretius", "republican")]

    augustan = [("livy", "augustan"), ("ovid", "augustan"), ("horace", "augustan"), ("vergil", "augustan"),
                ("hyginus", "augustan")]

    early_silver = [("martial", "early_silver"), ("juvenal", "early_silver"), ("tacitus", "early_silver"),
                    ("lucan", "early_silver"), ("quintilian", "early_silver"), ("sen", "early_silver"),
                    ("statius", "early_silver"), ("silius", "early_silver"), ("columella", "early_silver")]

    authors = republican + augustan + early_silver

    print("Authors:", authors)
    # authors = [("cicero", "republican"), ("caesar", "republican"), ("nepos", "republican")]

    text_df = get_text_by_authors(authors)
    # text_df.to_csv("./text_df/text-rep_aug.csv", index=False)
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

    save_model(bow_transformer, "bow_transformer_2")
    # print("DONE")
    # return

    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_train = bow_transformer.transform(X_train)  # ONLY TRAINING DATA

    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_test = bow_transformer.transform(X_test)  # TEST DATA

    return text_bow_train, text_bow_test, y_train, y_test


def save_model(model, filename):
    with open(filename + ".pkl", "wb") as fid:
        cPickle.dump(model, fid)


def load_model(filename):
    with open(filename + ".pkl", "rb") as fid:
        model = cPickle.load(fid)
        return model


def classify_text(text):
    model_filename = "./saved_models/SVC-testA_2"
    model = load_model(model_filename)

    text = remove_latin_library_items(preprocess(text))
    print(text)
    chunks = split_count(text, SPLIT_LENGTH)

    results = []

    bow_transformer_filename = "bow_transformer_2"
    bow_transformer = load_model(bow_transformer_filename)

    chunks = bow_transformer.transform(chunks)
    for chunk in chunks:
        cur_pred = model.predict_proba(chunk)
        results.append(cur_pred)

    results = np.squeeze(np.asarray(results))
    print("results", results, results.shape)
    averages = results.mean(0)
    print("averages", averages)
    return averages


def test_classify_text():
    text = "ruerint Danai, quaeque ipse miserrima vidi 5 et quorum pars magna fui. quis talia fando Myrmidonum Dolopumve aut duri miles Ulixi temperet a lacrimis? et iam nox umida caelo praecipitat suadentque cadentia sidera somnos. sed si tantus amor casus cognoscere nostros               10 et breviter Troiae supremum audire laborem, quamquam animus meminisse horret luctuque refugit, incipiam. fracti bello fatisque repulsi ductores Danaum tot iam labentibus annis instar montis equum divina Palladis arte               15 aedificant, sectaque intexunt abiete costas; votum pro reditu simulant; ea fama vagatur. huc delecta virum sortiti corpora furtim includunt caeco lateri penitusque cavernas ingentis uterumque armato milite complent.               20 est in conspectu Tenedos, notissima fama insula, dives opum Priami dum regna manebant, nunc tantum sinus et statio male fida carinis: huc se provecti deserto in litore condunt; nos abiisse rati et vento petiisse Mycenas.               25 ergo omnis longo soluit se Teucria luctu; panduntur portae, iuvat ire et Dorica castra desertosque videre locos litusque relictum: hic Dolopum manus, hic saevus tendebat Achilles; classibus hic locus, hic acie certare solebant.               30 pars stupet innuptae donum exitiale Minervae et molem mirantur equi; primusque Thymoetes duci intra muros hortatur et arce locari, sive dolo seu iam Troiae sic fata ferebant. at Capys, et quorum melior sententia menti,               35 aut pelago Danaum insidias suspectaque dona praecipitare iubent subiectisque urere flammis, aut terebrare cavas uteri et temptare latebras. scinditur incertum studia in contraria vulgus. Primus ibi ante omnis magna comitante caterv"
    classify_text(text)


def main():
    text_bow_train, text_bow_test, y_train, y_test = load_data()

    # y_train_hot = np.zeros((y_train.size, y_train.max() + 1))
    # y_train_hot[np.arange(y_train.size), y_train] = 1
    #
    # y_test_hot = np.zeros((y_test.size, y_train.max() + 1))
    # y_test_hot[np.arange(y_test.size), y_test] = 1

    # y_train = y_test_hot
    # y_test = y_test_hot

    # print(y_test)

    # Prints numbers of each class
    counts = np.bincount(y_test)
    ii = np.nonzero(counts)[0]
    print(list(zip(ii, counts[ii])))

    # print(y_test[0])
    # print(text_bow_test[0])

    # model = MultinomialNB()
    # model = LinearSVC()
    model = SVC(probability=True)

    model = model.fit(text_bow_train, y_train)
    print("Train Accuracy")
    print(model.score(text_bow_train, y_train))

    print("Eval Accuracy")

    print(model.score(text_bow_test, y_test))

    pred = model.predict_proba(text_bow_test[0])
    print(pred)
    save_model(model, "./saved_models/SVC-testA_2")


if __name__ == "__main__":
    # main()
    test_classify_text()
