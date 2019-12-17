import os
import roman
import re


def load_text(filename):
    file = open(filename, "r")
    text = file.read()
    return text


def save_text(filename, text):
    try:
        os.makedirs(os.path.dirname(filename))
    except FileExistsError:
        pass
    file = open(filename, "w+")
    file.write(text)


def is_number(string):  # checks if string is number or Roman Numeral
    if string.isdigit():
        return True
    try:
        roman.fromRoman(string)
    except roman.InvalidRomanNumeralError:
        return False
    return True


def clean_text(filename):
    text = load_text(filename)
    text = re.sub(r'\W+', ' ', text).strip("_")
    text = ''.join([word for word in text if not is_number(word)])
    text = text.lower()
    text = text[70:-20]
    write_filename = "./cleaned_texts/" + "/".join(filename.split("/")[2:])
    save_text(write_filename, text)
    print(write_filename)

    # print(text)


def clean_all_texts(texts_dir="./texts"):
    authors_dirs = os.listdir(texts_dir)
    for authors_dir in authors_dirs:
        if authors_dir in [".DS_Store"]:
            continue
        author_texts = os.listdir(texts_dir + "/" + authors_dir)
        for author_text in author_texts:
            author_text_filename = texts_dir + "/" + authors_dir + "/" + author_text
            # print("author text", author_text_filename)
            clean_text(author_text_filename)


if __name__ == "__main__":
    clean_all_texts()
