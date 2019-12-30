from cltk.corpus.utils.importer import CorpusImporter
from cltk.corpus.readers import get_corpus_reader
from cltk.corpus.latin.latin_library_corpus_types import corpus_texts_by_type
from cltk.corpus.readers import assemble_corpus, get_corpus_reader
from cltk.corpus.latin.latin_library_corpus_types import corpus_texts_by_type, corpus_directories_by_type


def download():
    corpus_importer = CorpusImporter('latin')
    corpus_importer.import_corpus('latin_text_latin_library')
    print(corpus_importer.list_corpora)


def playground():
    periods = ['republican', 'augustan', 'early_silver', 'late_silver']
    print(list(corpus_texts_by_type.values())[:2])
    # reader = get_corpus_reader(language='latin', corpus_name='sall.1.txt')
    latin_corpus = get_corpus_reader(corpus_name='latin_text_latin_library', language='latin')
    # filtered_reader, fileids, categories = assemble_corpus(latin_corpus, types_requested=periods,
    #                                                        type_dirs=corpus_directories_by_type,
    #                                                        type_files=corpus_texts_by_type)
    # print(len(list(filtered_reader.docs())))
    print(len((list(latin_corpus.docs()))))
    # sentences = list(latin_corpus.sents())
    # print(len(sentences))

    # print("num docs", len(list(latin_corpus.docs())))
    # print("num paras", len(list(latin_corpus.paras())))


if __name__ == "__main__":
    # download()
    playground()
    # my_latin_downloader = CorpusImporter('latin')
    #
    # my_latin_downloader.import_corpus('latin_text_latin_library')
    # my_latin_downloader.import_corpus('latin_models_cltk')
