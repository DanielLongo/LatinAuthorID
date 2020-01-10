from cltk.corpus.readers import assemble_corpus, get_corpus_reader
from cltk.corpus.latin.latin_library_corpus_types import corpus_texts_by_type, corpus_directories_by_type

from cltk.corpus.utils.importer import CorpusImporter
from cltk.stem.latin.j_v import JVReplacer
from cltk.tokenize.word import WordTokenizer
from cltk.stem.lemma import LemmaReplacer
import re


def remove_numbers(text):
    for i in range(10):
        text = text.replace(str(i), "")
    return text


def remove_latin_library_items(text):
    # text : String
    assert (type(text) == str), "text must be a string"
    # regex = re.compile(".*?\((.*?)\)")
    # text = re.findall(regex, text)
    brackets_removed = re.sub("[\(\[].*?[\)\]]", "", text)
    # print("+" * 50, brackets_removed, "=" * 50)
    digits_removed = remove_numbers(brackets_removed)
    dash_removed = digits_removed.replace("-", "")
    split = dash_removed.split(" ")
    cropped = split[15:-15]
    joined = " ".join(cropped)
    return joined
    # text = text.strip()


def preprocess(doc):
    assert (type(doc) == str)
    word_tokenizer = WordTokenizer('latin')
    doc_word_tokens = word_tokenizer.tokenize(doc)
    doc_word_tokens_no_punt = [token.lower() for token in doc_word_tokens if token not in ['.', ',', ':', ';']]

    # lemmeatization
    corpus_importer = CorpusImporter('latin')
    corpus_importer.import_corpus('latin_models_cltk')
    jv_replacer = JVReplacer()

    lemmatizer = LemmaReplacer('latin')
    lemmata = lemmatizer.lemmatize(" ".join(doc_word_tokens_no_punt))
    cleaned = remove_latin_library_items(" ".join(lemmata))
    return cleaned


def load_docs(types_requested):
    latin_corpus = get_corpus_reader(corpus_name='latin_text_latin_library', language='latin')
    reader = assemble_corpus(latin_corpus, types_requested=types_requested, type_dirs=corpus_directories_by_type,
                             type_files=corpus_texts_by_type)
    docs = list(reader.docs())
    return docs
