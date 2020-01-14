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

# AUTHORS = ["cicero", "caesar", "nepos", "lucretius", "livy", "ovid", "horace", "vergil", "hyginus", "martial",
#            "juvenal", "tacitus", "lucan", "quintilian", "sen", "statius", "silius", "columella"]
AUTHORS = ['caesar', 'cicero', 'columella', 'horace', 'hyginus', 'juvenal', 'livy', 'lucan', 'lucretius', 'martial', 'nepos',
 'ovid', 'quintilian', 'sen', 'silius', 'statius', 'tacitus', 'vergil']

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
    authors = sorted(authors, key=lambda x: x[0])

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

    save_model(bow_transformer, "bow_transformer_3")
    # print("DONE")
    # return

    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_train = bow_transformer.transform(X_train)  # ONLY TRAINING DATA

    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_test = bow_transformer.transform(X_test)  # TEST DATA

    return text_bow_train, text_bow_test, y_train, y_test


def save_model(model, filename):
    print("not saving")
    # with open(filename + ".pkl", "wb") as fid:
    #     cPickle.dump(model, fid)


def load_model(filename):
    with open(filename + ".pkl", "rb") as fid:
        model = cPickle.load(fid)
        return model


def classify_text(text):
    model_filename = "./saved_models/SVC-testA_2"
    model = load_model(model_filename)

    print("TEXT A", text)
    text = remove_latin_library_items(preprocess(text))
    print("TEXT B", text)
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
    print("authors", AUTHORS)
    return averages


def test_classify_text():
    # text = "Me perlongo intervallo prope memoriae temporumque nostrorum primum hominem novum consulem fecistis et eum locum quem nobilitas praesidiis firmatum atque omni ratione obvallatum tenebat me duce rescidistis virtutique in posterum patere voluistis. Neque me tantum modo consulem, quod est ipsum per sese amplissimum, sed ita fecistis quo modo pauci nobiles in hac civitate consules facti sunt, novus ante me nemo. Nam profecto, si recordari volueritis de novis hominibus, reperietis eos qui sine repulsa consules facti sunt diuturno labore atque aliqua occasione esse factos, cum multis annis post petissent quam praetores fuissent, aliquanto serius quam per aetatem ac per leges liceret; qui autem anno suo petierint, sine repulsa non esse factos; me esse unum ex omnibus novis hominibus de quibus meminisse possimus, qui consulatum petierim cum primum licitum sit, consul factus sim cum primum petierim, ut vester honos ad mei temporis diem petitus, non ad alienae petitionis occasionem interceptus, nec diuturnis precibus efflagitatus, sed dignitate impetratus esse videatur. [4] Est illud amplissimum quod paulo ante commemoravi, Quirites, quod hoc honore ex novis hominibus primum me multis post annis adfecistis, quod prima petitione, quod anno meo, sed tamen magnificentius atque ornatius esse illo nihil potest, quod meis comitiis non tabellam vindicem tacitae libertatis, sed vocem vivam prae vobis indicem vestrarum erga me voluntatum ac studiorum tulistis. Itaque me non extrema diribitio suffragiorum, sed primi illi vestri concursus, neque singulae voces praeconum, sed una vox universi populi Romani consulem declaravit. [5] Hoc ego tam insigne, tam singulare vestrum beneficium, Quirites, cum ad animi mei fructum atque laetitiam duco esse permagnum, tum ad curam sollicitudinemque multo magis. Versantur enim, Quirites, in animo meo multae et graves cogitationes quae mihi nullam partem neque diurnae neque nocturnae quietis impertiunt, primum tuendi consulatus, quae cum omnibus est difficilis et magna ratio, tum vero mihi praeter ceteros cuius errato nulla venia, recte facto exigua laus et ab invitis expressa proponitur; non dubitanti fidele consilium, non laboranti certum subsidium nobilitatis ostenditur. [6] Quod si solus in discrimen aliquod adducerer, ferrem, Quirites, animo aequiore; sed mihi videntur certi homines, si qua in re me non modo consilio verum etiam casu lapsum esse arbitrabuntur, vos universos qui me antetuleritis nobilitati vituperaturi. Mihi autem, Quirites, omnia potius perpetienda esse duco quam non ita gerendum consulatum ut in omnibus meis factis atque consiliis vestrum de me factum consiliumque laudetur. Accedit etiam ille mihi summus labor ac difficillima ratio consulatus gerendi, quod non eadem mihi qua superioribus consulibus lege et condicione utendum esse decrevi, qui aditum huius loci conspectumque vestrum partim magno opere fugerunt, partim non vehementer secuti sunt. Ego autem non solum hoc in loco dicam ubi est id dictu facillimum, sed in ipso senatu in quo esse locus huic voci non videbatur popularem me futurum esse consulem prima illa mea oratione Kalendis Ianuariis dixi. [7] Neque enim ullo modo facere possum ut, cum me intellegam non hominum potentium studio, non excellentibus gratiis paucorum, sed universi populi Romani iudicio consulem ita factum ut nobilissimis hominibus longe praeponerer, non et in hoc magistratu et in omni vita <videar> esse popularis. Sed mihi ad huius <verbi> vim et interpretationem vehementer opus est vestra sapientia. Versatur enim magnus error propter insidiosas non nullorum simulationes qui, cum populi non solum commoda verum etiam salutem oppugnant et impediunt, oratione adsequi volunt ut populares esse videantur. [8] Ego qualem Kalendis Ianuariis acceperim rem publicam, Quirites, intellego, plenam sollicitudinis, plenam timoris; in qua nihil erat mali, nihil adversi quod non boni metuerent, improbi exspectarent; omnia turbulenta consilia contra hunc rei publicae statum et contra vestrum otium partim iniri, partim nobis consulibus designatis inita esse dicebantur; sublata erat de foro fides non ictu aliquo novae calamitatis, sed suspicione ac perturbatione iudicio"
    text = "[1] Est hoc in more positum, Quirites, institutoque maiorum, ut ei qui beneficio vestro imagines familiae suae consecuti sunt eam primam habeant contionem, qua gratiam benefici vestri cum suorum laude coniungant."
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
    save_model(model, "./saved_models/SVC-testA_3")


if __name__ == "__main__":
    # main()
    test_classify_text()
