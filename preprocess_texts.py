from cltk.tokenize.sentence import TokenizeSentence
# from cltk.tokenize.word import nltk_tokenize_words

# tokenizer = TokenizeSentence('latin')
def load_text(filename):
    file = open(filename, "r")
    text = file.readlines()
    return text

# nltk_tokenize_words(

if __name__ == "__main__":
    filename = "cleaned_texts/cicero/legagr2.txt"
    text = load_text(filename)
    print(text)
    print("Finished")