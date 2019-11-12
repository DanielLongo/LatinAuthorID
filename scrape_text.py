import requests
import re
import os
import errno
import httplib2
from bs4 import BeautifulSoup, SoupStrainer

skip_links = ["https://www.thelatinlibrary.com/addison.html"]
already_downloaded = []
def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

def download_all(url):
    # try:
    global already_downloaded
    if url in already_downloaded:
        print("already downloaded")
        return
    else:
        already_downloaded.append(url)
    print("url", url)
    # if url in skip_links:
    #     return
    if url.split(".")[-1] == "shtml":
        print("DOWNLOADING")
        download_from_url(url)
    else:
        links, is_text_page = get_all_hyperlinks(url)
        for link in links:
            download_all(link)

def get_all_hyperlinks(url):
    http = httplib2.Http()
    status, response = http.request(url)

    links = []
    in_text_page = True
    for link in BeautifulSoup(response, parse_only=SoupStrainer('a'), features="html.parser"):
        if link.has_attr('href'):
            if link["href"].split(".")[-1] == "shtml":
                in_text_page = False # text pages have no hyperlinks with .shtml

            if (link["href"]).strip("/") != "index.html" and (link["href"]).strip("/") != "classics.html":
                print(link["href"])
                cur_link = "/".join(url.split("/")[:-1] + [link["href"]])
                links.append(cur_link)
            # print(link['href'])
    return links[:-1], in_text_page

def remove_with_nums(text):
    new = re.sub(r'\w*\d\w*', '', text).strip()
    return new


def remove_brackets(text):
    text = text.strip("[")
    text = text.strip("]")
    return text.strip("[]")


def scrape_text_page(url):
    file = requests.get(url)
    file_text = file.text
    return file_text


def save_text(text, filename):
    # create new dir if necessary
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise
    file = open(filename, "w+")
    file.write(text)
    file.close()


def get_filename(url):
    # names file by end of url
    # for example https://www.thelatinlibrary.com/cicero/legagr2.shtml want cicero/legar2
    # folder_name = url.split("/")[-2]
    if url.split(".")[-1] != "shtml":
        print("url wrong ending", url)
        return None
    last_part = url.split("/")[-2] + "/" + url.split("/")[-1]
    filename = last_part.replace(".shtml", "")
    filename = filename.replace(".", "-") + ".txt"
    # filename = last_part.split(".")[0] + ".txt"  # replace .shtml with .txt
    return filename


def download_from_url(url):
    text = scrape_text_page(url)
    text = clean_html(text)
    # text = remove_brackets(text)
    # text = remove_with_nums(text)
    text = " ".join(text.split())
    filename = get_filename(url)
    if filename is None:
        return
    filename = "texts/" + filename
    print("filename:", filename)
    save_text(text, filename)


if __name__ == "__main__":
    # url = "https://www.thelatinlibrary.com/cicero/legagr2.shtml"
    # download_from_url(url)
    # url = "https://www.thelatinlibrary.com/cic.html"
    # url = "https://www.thelatinlibrary.com/cicero/quinc.shtml"
    url = "https://www.thelatinlibrary.com/indices.html"
    # url = "https://www.thelatinlibrary.com/alcuin/rhetorica.shtml"
    links, _ = get_all_hyperlinks(url)
    for link in links:
        print("Link", link)
        download_all(link)

    # text = scrape_text_page(url)
    # text = clean_html(text)
    # # text = remove_brackets(text)
    # # text = remove_with_nums(text)
    # text = " ".join(text.split())
    # filename = "texts/" + get_filename(url)
    # save_text(text, filename)
    # text = text.strip("\n")

    # print(type(text))
    # print(text)
