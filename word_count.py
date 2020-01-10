import inspect

from cltk.corpus.utils.importer import CorpusImporter
from cltk.corpus.readers import get_corpus_reader
from cltk.corpus.latin.latin_library_corpus_types import corpus_texts_by_type
from cltk.corpus.readers import assemble_corpus, get_corpus_reader
from cltk.corpus.latin.latin_library_corpus_types import corpus_texts_by_type, corpus_directories_by_type

from cltk.prosody.latin.scanner import Scansion
from cltk.prosody.latin.clausulae_analysis import Clausulae
from cltk.tokenize.word import WordTokenizer
from collections import Counter
from cltk.corpus.utils.importer import CorpusImporter
from cltk.stem.latin.j_v import JVReplacer
from cltk.tokenize.word import WordTokenizer
from cltk.stem.lemma import LemmaReplacer
import re
from string import digits
import pandas as pd
import spacy
import scattertext as st
from IPython.display import IFrame
from IPython.core.display import display, HTML
from scattertext import CorpusFromPandas, produce_scattertext_explorer

from utils import *

display(HTML("&lt;style>.container { width:98% !important; }&lt;/style>"))
nlp = spacy.load('en')


def get_word_count_for_period(period):
    docs = load_docs([period])
    cleaned_docs = []
    for doc in docs:
        cleaned_docs.append(preprocess(doc))
    counts = Counter({})
    for cleaned_doc in cleaned_docs:
        counts += Counter(cleaned_doc.split(" "))
    return counts


def get_df_for_period(period):
    texts_df = pd.DataFrame(columns=["period", "text"])
    i = 0
    docs = load_docs([period])
    cleaned_docs = []
    stop_early = -1
    z = 0
    for doc in docs:
        z += 1
        cleaned_docs.append(remove_latin_library_items(preprocess(doc)))
        if z == stop_early:
            break
    rows = []
    for i, cleaned_doc in enumerate(cleaned_docs):
        cur_doc = {"period": period, "text": cleaned_doc}
        rows.append(cur_doc)
        # texts_df[i] = [period, cleaned_doc]
        # print(i, cleaned_doc)
    texts_df = pd.DataFrame(rows)
    return texts_df


def get_words_counts():
    all_periods = corpus_directories_by_type.keys()
    all_periods = ['republican', 'augustan']
    counts = {}
    frames = []
    for period in all_periods:
        # counts[period] = get_word_count_for_period(period)
        cur_df = get_df_for_period(period)
        frames.append(cur_df)
        # break
    combined_frames = pd.concat(frames)
    print("_+_" * 300)
    print("CUR", cur_df)
    print("COMBINED", combined_frames)
    print("_*_" * 300)
    corpus = st.CorpusFromPandas(combined_frames,
                                 category_col='period',
                                 text_col='text',
                                 nlp=nlp).build()

    html = produce_scattertext_explorer(corpus,
                                        category='republican',
                                        category_name='Republican',
                                        not_category_name='Augustan',
                                        width_in_pixels=1000,
                                        minimum_term_frequency=20,
                                        transform=st.Scalers.scale)
    # metadata=convention_df['speaker'])
    file_name = './figures/testing-4.html'
    open(file_name, 'wb').write(html.encode('utf-8'))
    IFrame(src=file_name, width=1200, height=700)
    print(cur_df)


def playground():
    print(get_words_counts())
    # periods = ['republican', 'augustan', 'early_silver', 'late_silver']
    # print(corpus_directories_by_type.keys())
    # print(corpus_directories_by_type.values())
    # print(list(corpus_texts_by_type.values()))
    # print(list(corpus_texts_by_type.keys()))
    # print(list(corpus_texts_by_type.values())[:2])
    # reader = get_corpus_reader(language='latin', corpus_name='sall.1.txt')
    # latin_corpus = get_corpus_reader(corpus_name='latin_text_latin_library', language='latin')
    # all_periods = corpus_directories_by_type.keys()
    # for period in all_periods:
    #     cur_docs = load_docs(types_requested=[period])
    #     print(cur_docs[0])
    #
    # docs = load_docs(types_requested=['republican', 'augustan'])
    # print(len(docs))
    # # print(docs[0])
    # return
    # reader = assemble_corpus(latin_corpus, types_requested=['republican', 'augustan'],
    #                          type_dirs=corpus_directories_by_type, type_files=corpus_texts_by_type)
    # doc = list(reader.docs())[0]

    # s = Scansion()
    # c = Clausulae()
    # s.scan_text(doc)

    # need jv replace https://github.com/cltk/tutorials/blob/master/4%20Lemmatization.ipynb

    # word_tokenizer = WordTokenizer('latin')
    # doc_word_tokens = word_tokenizer.tokenize(doc)
    # doc_word_tokens_no_punt = [token.lower() for token in doc_word_tokens if token not in ['.', ',', ':', ';']]
    # # print(doc_word_tokens_no_punt)
    # doc_counts_counter = Counter(doc_word_tokens_no_punt)
    # # print(doc_counts_counter)
    #
    # # print(list(reader.docs())[0])
    # # print(inspect.getmodule(type(z)))
    # # print(inspect.getmodule(type(z.paras())))
    # # print("kfhjdsklfh", len(list(filtered_reader.docs())))
    #
    # # filtered_reader, fileids, categories = assemble_corpus(latin_corpus, types_requested=periods,
    # #                                                        type_dirs=corpus_directories_by_type,
    # #                                                        type_files=corpus_texts_by_type)
    # # print(len(list(filtered_reader.docs())))
    # print(len((list(latin_corpus.docs()))))
    # sentences = list(latin_corpus.sents())
    # print(len(sentences))

    # print("num docs", len(list(latin_corpus.docs())))
    # print("num paras", len(list(latin_corpus.paras())))

    # # lemmeatization
    # corpus_importer = CorpusImporter('latin')
    # corpus_importer.import_corpus('latin_models_cltk')
    # jv_replacer = JVReplacer()
    #
    # lemmatizer = LemmaReplacer('latin')
    # # print("tokens", doc_word_tokens[-1])
    # lemmata = lemmatizer.lemmatize(" ".join(doc_word_tokens_no_punt))
    # # print("ORIGINAL", doc[:])
    # # print("_" * 1000)
    # # print("NEW", " ".join(lemmata)[:])
    # # print("*" * 1000)
    # cleaned = remove_latin_library_items(" ".join(lemmata))
    # # print("CLEANED", cleaned)
    # print(Counter(cleaned.split(" ")))


if __name__ == "__main__":
    playground()
