"""
@author: Preetham Salehundam
@email: salehundam.2@wright.edu

this file augments the precomputed features using pyspark and add wordcount and size of article
"""

import pandas as pd
import requests
import sys


def page_search(session, title):
    """
    :param session: http session from wikipedia API
    :param title: find the page with wiki title
    :return: title, size, word_count, articleid
    """
    url = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": title
    }
    response = session.get(url=url, params=PARAMS).json()
    query = response["query"]
    search = query["search"]
    title = None
    if len(search) > 0:
        title = search[0]["title"]
        articleid = int(search[0]["pageid"])
        size = int(search[0]["size"])
        word_count = int(search[0]["wordcount"])
    elif "suggestion" in query.get("searchinfo", []):
        title = query["searchinfo"]["suggestion"]
        size = 0
        word_count = 0
        articleid = 0
    else:
        print("title not found!")
        print(title, query)
        size, word_count, articleid = None, None, None

    return title, size, word_count, articleid


def fetch_categories(session):
    """
    :param session: http session for wiki api
    :return: closure which fetches categories of article title
    """
    def __fetch_categories(row):
        URL = "https://en.wikipedia.org/w/api.php"

        PARAMS = {
            "action": "query",
            "format": "json",
            "prop": "categories",
            "pageids": row["articleid"]
        }

        R = session.get(url=URL, params=PARAMS)
        DATA = R.json()

        PAGES = DATA["query"]["pages"]

        for k, v in PAGES.items():
            try:
                if 'categories' in v:
                    for cat in v['categories']:
                        row["category"] = cat["title"]
                        print(row.articletitle, "==>", cat["title"])
                        break
            except Exception as err:
                print(err)
        return row
    return __fetch_categories




def DFS(session, article_title, size, word_count):
    """
    :param session: http session for wiki api
    :param article_title: title of wiki article
    :param size: article size
    :param word_count: word count
    :return: None
    """
    revisions_url = "https://en.wikipedia.org/w/api.php" #?action=query&prop=revisions&format=json&&rvlimit=5&formatversion=2".format(article_title)
    PARAMS = {
        "action": "query",
        "format": "json",
        "prop": "revisions",
        "rvlimit": 5,
        "formatversion":2,
        "titles": article_title
    }
    diff_url = "http://en.wikipedia.org/w/index.php?diff={}&oldid={}"
    response= session.get(url=revisions_url, params=PARAMS).json()
    pages = response["query"]["pages"][0]
    df = pd.DataFrame()
    try:
        revisions = pages["revisions"]
        if size is not None and word_count is not None:
            title, size, word_count, articleid = page_search(session, article_title)
    except KeyError as err:
        # HARD LUCK
        # =================================
        # spell mistakes of article title
        # query wikipedia for corrected title
        # print(pages)
        # url = "https://en.wikipedia.org/w/api.php"
        # PARAMS = {
        #     "action": "query",
        #     "format": "json",
        #     "list": "search",
        #     "srsearch": pages["title"]
        # }
        # response = session.get(url=url, params=PARAMS).json()
        # query = response["query"]
        # search = query["search"]
        # title = None
        # if len(search) > 0:
        #     title = search[0]["title"]
        # elif "suggestion" in query.get("searchinfo",[]):
        #     title= query["searchinfo"]["suggestion"]
        # else:
        #     print("title not found!")
        #     print(query)
        # ====================================
        title, size, word_count, articleid = page_search(session=session, title=pages["title"])
        if title is None:
            return df
        print(pages["title"], " ==> ", title)
        df = DFS(session, title, size, word_count)
        return df

    df = df.append({"size":size, "word_count":word_count, "article_title":title, "articleid": int(articleid)}, ignore_index=True)

    return df


def augment_data():
    """
    :return:filenames of collected edits
    """
    sess = requests.Session()
    df = pd.DataFrame()
    edits = pd.read_csv("edits.csv")
    edits_df = edits
    #.sample(n=10)  # [["editid", "editor", "newrevisionid", "oldrevisionid", "editcomment", "diffurl", "articletitle"]]
    try:
        for index, row in edits_df.iterrows():
            _df=DFS(sess,row["articletitle"], size=0, word_count=0)
            if not _df.empty:
                edits_df.loc[index, "size"] = _df["size"][0]
                edits_df.loc[index, "word_count"] = _df["word_count"][0]
                edits_df.loc[index, "articleid"] = _df["articleid"][0]
                edits_df.loc[index, "articletitle"] = _df["article_title"][0]
            else:
                print("failed")
                print(row["articletitle"])
        filename = "augmented_with_size_wc.csv"
        edits_df = edits_df.dropna()
        edits_df.to_csv(filename, index=False)
        print(edits_df.count())
    except Exception as err:
        filename = "incomplete.csv"
        edits_df.to_csv(filename, index=False)
        print(edits_df.count())
        print(err)
        sys.exit(0)
    except KeyboardInterrupt as err:
        filename = "incomplete.csv"
        edits_df.to_csv(filename, index=False)
        print(edits_df.count())
        print(err)
        sys.exit(0)
    return filename

if __name__ == "__main__":
    merged_data = pd.read_csv("merged_augmented.csv")
    merged_data = merged_data.drop(["character_distribution"], axis=1)
    merged_data = merged_data.fillna(0)
    sess = requests.session()
    merged_data = merged_data.apply(fetch_categories(session=sess), axis=1)
    merged_data.to_csv("category_data.csv", index=False)
    print("Augmentation complete!!")