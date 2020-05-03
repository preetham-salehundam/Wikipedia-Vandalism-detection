from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.functions import input_file_name
from pyspark.sql.types import Row, StringType, IntegerType, FloatType, LongType, StructField, StructType
import os, shutil, glob, time
from bs4 import BeautifulSoup
import requests
from collections import Counter
import numpy as np

special_tokens = "|".join(
    [".", ",", ":", ";", "«", "»", "’", "|", "?", "!", "=", "(", ")", "*", "[[", "]]", "[. ]", "{{", "}}", "{", "}"])

sent = "hi this is . preetham , checking  hhhhkh hkhjkhh  hhjhh hkhjkhk hhjhjkh hkkhkh"


def remove_special_chars(data):
    special_tokens = "".join(
        [".", ",", ":", ";", "«", "»", "’", "|", "?", "!", "=", "(", ")", "*", "[[", "]]", "[. ]", "{{", "}}", "{",
         "}"])
    table = str.maketrans(special_tokens, " " * len(special_tokens))
    return "".join(list(filter(lambda x: len(x) > 0, data.translate(table).split(" "))))


def upper_to_lower_ratio(data_rdd):
    upper_count = data_rdd.filter(lambda x: x.isupper()).map(lambda x: ("upper", 1)).reduceByKey(
        lambda x, y: x + y).collect()
    lower_count = data_rdd.filter(lambda x: x.islower()).map(lambda x: ("lower", 1)).reduceByKey(
        lambda x, y: x + y).collect()
    upper = upper_count[-1][-1] if len(upper_count) > 0 else 0
    lower = lower_count[-1][-1] if len(lower_count) > 0 else 0
    print(lower, upper)
    upper_to_lower = (1 + (upper)) / (1 + (lower))
    upper_to_all = (1 + (upper)) / (1 + (lower + upper))
    return upper_to_lower, upper_to_all


# @udf(FloatType())
def u_to_l_ratio(data):
    chars = list(data)
    upper_count = sum([char.isupper() for char in chars])
    lower_count = sum([char.islower() for char in chars])
    # if (upper_count==0 and lower_count ==0):
    #     return 0.0
    return round((1 + (upper_count)) / (1 + (lower_count)), 4)


def u_to_all_ratio(data):
    chars = list(data)
    upper_count = sum([char.isupper() for char in chars])
    lower_count = sum([char.islower() for char in chars])
    # if (upper_count==0 and lower_count ==0):
    #     return 0.0
    return round((1 + (upper_count)) / (1 + (lower_count) + (upper_count)), 4)


def char_diversity(data):
    """
    used to detect non-sense and random keyboard strokes
    :param data:
    :return char diversity score:
    """
    edit_length = len(data)
    if len(data) <= 0:
        return 0.0
    unique_chars = list(set(data))
    return round(edit_length ** (1 / len(unique_chars)), 4) if len(unique_chars)!=0 else 0


def digit_ratio(data):
    data = list(remove_special_chars(data))
    digit_len = len(list(filter(lambda x: x.isnumeric(), data)))
    return round((1 + digit_len) / (1 + len(data)), 4)


def non_alpha_ratio(data):
    data = list(remove_special_chars(data))
    non_alpha_chars = len(list(filter(lambda x: not x.isalpha(), data)))
    return round((1 + non_alpha_chars) / (1 + len(data)), 4)


basename_udf = udf(lambda z: os.path.basename(z), StringType())
comment_len_udf = udf(lambda x: len(x) if x != "null" and x is not None else 0, IntegerType())
# if none consider anonymous
anonymous_check_udf = udf(lambda x: int(not (x.isalnum() or x.isalpha())) if x is not None else 1, IntegerType())

# class encoder

class_encoder_udf = udf(lambda x: 0 if x == "regular" else 1, IntegerType())

# requests
# def get_edit_content(x, http_session,mode="added"):
#     if x is not None:
#         try:
#             x = http_session.get(x).text
#         except Exception as e:
#             print(e)
#         #print("mode {} \n html {}".format(mode, x))
#         added_content = BeautifulSoup(x).body.find("td", attrs={"class": "diff-addedline"})
#         deleted_content = BeautifulSoup(x).body.find("td", attrs={"class": "diff-deletedline"})
#         added_content = added_content.get_text() if added_content is not None else ""
#         deleted_content = deleted_content.get_text() if deleted_content is not None else ""
#
#         #print("mode {} \n content {}".format(mode, content))
#     else:
#         added_content = ""
#         deleted_content = ""
#     return abs(len(added_content) - len(deleted_content))


schema = StructType([
    StructField('added_content', StringType()),
    StructField('deleted_content', StringType())
])


def get_edit_content(x, http_session, mode="added"):
    #print(x)
    if x is not None:
        try:
            x = http_session.get(x.replace("http","https")).text
            time.sleep(0.5)
        except Exception as e:
            print(e)
        # print("mode {} \n html {}".format(mode, x))
        added_content = BeautifulSoup(x).body.find("td", attrs={"class": "diff-addedline"})
        deleted_content = BeautifulSoup(x).body.find("td", attrs={"class": "diff-deletedline"})
        added_content = added_content.get_text() if added_content is not None else ""
        deleted_content = deleted_content.get_text() if deleted_content is not None else ""

        # print("mode {} \n content {}".format(mode, content))
    else:
        added_content = ""
        deleted_content = ""
    return added_content, deleted_content
    # return abs(len(added_content) - len(deleted_content))


def size_increment(added_content, deleted_content):
    return abs(len(added_content) - len(deleted_content))


size_increment_udf = udf(lambda x, y: size_increment(x, y))

# html_parser_udf = udf(lambda x: get_edit_content(x, session), schema)
html_parser_udf = udf(lambda x: get_edit_content(x, session), schema)

# for efficient search
vulgars_dict = []
with open('vulgar_corpus.txt') as fd:
    vulgars_dict = {K.lower().strip(): None for K in fd.read().split()}

contractions = []
with open("contractions.txt") as fd:
    contractions = {K.lower().strip(): None for K in fd.read().split()}


def contraction_word_count(edit):
    words = [1 if word.lower() in contractions else 0 for word in edit.split()]
    contraction_count = sum(words)
    return contraction_count / len(words) if len(words) != 0 else 0


bad_words_info = udf(lambda x: contraction_word_count(x))


def vulgar_word_ratio(edit):

    words = [1 if word.lower() in vulgars_dict else 0 for word in edit.split()]
    vulgar_count = sum(words)
    #print(edit, vulgar_count)
    return vulgar_count / len(words) if len(words) != 0 else 0


vulgar_word_info = udf(lambda x: vulgar_word_ratio(x))

longest_varible = udf(lambda x: max([len(word) for word in x.split()]) if x is not None and len(x.split())>0 else 0)


# def average_term_frequency(edit):
#     words_dict = {word.lower():None for word in edit.split()}

def kl_divergence(p, q):
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def character_distribution(old, new):
    # print("old: ", old)
    # print("new: ", new)
    # print('===' * 5)
    counts_old = Counter(old)
    counts_new = Counter(new)
    total_old, total_new = sum(counts_old.values()), sum(counts_new.values())
    dist_old = []
    dist_new = []
    for val in counts_old.values():
        dist_old.append(val / total_old)

    for val in counts_new.values():
        dist_new.append(val / total_new)
    #print("char dist")

    padd = len(dist_old) - len(dist_new)
    if padd > 0:
        dist_new += [0] * abs(padd)
    else:
        dist_old += [0] * abs(padd)
    #print(dist_old, dist_new)
    return float(kl_divergence(np.array(dist_new), np.array(dist_old)))


character_distribution_udf = udf(lambda x, y: character_distribution(x, y) if x is not None and y is not None else 0.0,
                                 FloatType())

# vulgarism_frequency = lamnda x:

# def get_file_names(data_rdd):

if __name__ == '__main__':
    for filename in """
augment
    """.split():
        sc = SparkSession.builder.appName("data-preprocess").getOrCreate()
        session = requests.Session()
        # print("processing ", filename)
        # lines = sc.read.text("/Users/preetham/Downloads/"
        #                       "pan10-wikipedia-vandalism-detection-training-corpus-2010-03-15/article-revisions/part05/327330256.txt").rdd.map(lambda x: x[0])
        # filename="part16"
        lines = sc.read.text("wikipedia-corpus/article-revisions/"+filename+"/*.txt").select(
            input_file_name(), "value").rdd.reduceByKey(lambda x, y: x + " " + y).map(
            lambda x: Row(oldrevisionid=os.path.basename(x[0]).replace(".txt", ""),
                          value=remove_special_chars(x[1]),
                          upper_to_lower=u_to_l_ratio(x[1]),
                          upper_to_all=u_to_all_ratio(x[1]),
                          # comment_len=comment_len_udf(remove_special_chars(x[1])),
                          digit_ratio=digit_ratio(x[1]),
                          non_alpha_numeric_ratio=non_alpha_ratio(x[1]),
                          character_diversity=char_diversity(remove_special_chars(x[1])),
                          # character_dist=0,
                          # compressibility=0,
                          # size_increment=0,
                          # size_ratio=0
                          ))
        # edits = sc.read.csv("/Users/preetham/Downloads/"
        #                     "pan10-wikipedia-vandalism-detection-training-corpus-2010-03-15/augemented_edits.csv", header=True)
        edits = sc.read.csv("augmented_edits.csv", header=True)

        edits_df = edits.select(["editid", "editor", "newrevisionid", "oldrevisionid", "editcomment", "diffurl", "size", "word_count", "articletitle"])
        # print(edits_df.show())
        # print("Edits - {}".format(edits_df.count()))
        df = sc.createDataFrame(lines)
        print("data :{}".format(df.count()))

        # annotations = sc.read.csv("/Users/preetham/Downloads/"
        #                           "pan10-wikipedia-vandalism-detection-training-corpus-2010-03-15/gold-annotations_augmented.csv",
        #                           header=True)
        annotations = sc.read.csv("gold_annotations_augment.csv",
                                      header=True)
        annotations_df = annotations.select(["editid", "class"])
        annotations_df = annotations_df.withColumn("class", class_encoder_udf(annotations_df["class"])).withColumn(
            "editid", annotations_df.editid.cast(LongType()))
        # print(annotations_df.collect())
        edit_revisions_df = df.join(edits_df, on="oldrevisionid", how="left").join(annotations_df, on="editid",
                                                                                   how="inner")
        # print(edit_revisions_df.collect())
        edit_revisions_df = edit_revisions_df.withColumn("comment_len", comment_len_udf(edit_revisions_df.editcomment)) \
            .withColumn("Anonymous", anonymous_check_udf(edit_revisions_df.editor)) \
            .withColumn("edited_content", html_parser_udf(edit_revisions_df.diffurl))

        edit_revisions_df = edit_revisions_df.withColumn("diffurl", size_increment_udf(
            edit_revisions_df.edited_content.added_content, edit_revisions_df.edited_content.deleted_content)) \
            .withColumnRenamed("diffurl", "size_increment") \
            .withColumn("vulgar_word_ratio", vulgar_word_info(edit_revisions_df.edited_content.added_content)) \
            .withColumn("longest_Word", longest_varible(edit_revisions_df.edited_content.added_content)) \
            .withColumn("bad_words", bad_words_info(edit_revisions_df.edited_content.added_content))
            # .withColumn("character_distribution",
            #             character_distribution_udf(edit_revisions_df.edited_content.deleted_content,
            #                                        edit_revisions_df.edited_content.added_content)) \

            # .withColumn("added_words", edit_revisions_df.edited_content.added_content )\
            # .withColumn("deleted_words",edit_revisions_df.edited_content.deleted_content)
        # print(edit_revisions_df.select(["Anonymous", "editor"]).show())
        edit_revisions_df = edit_revisions_df.drop("value").drop("editcomment").drop("edited_content")
        #print(edit_revisions_df.collect())
        edit_revisions_df.repartition(1).write.csv("wiki_cleaned_csv", mode="overwrite", header=True)

        # print(edit_revisions_df.show())
        # print(df.show())

        # counts = lines.flatMap(lambda  x: x.split(' ')).filter().map(lambda x : (x , 1)).reduceByKey(lambda x,y : x+y)

        # counts = remove_special_chars(lines)
        # output = counts.collect()
        # print(output)

        # output = upper_to_lower_ratio(lines)
        # print(output)

        # for (word, count) in output:
        #     print("%s: %i" % (word, count))
        # lines.collect()

        # print(u_to_l_ratio("This is BullShit"))
        # os.copy("wiki_cleaned.csv/part-00000-e8de2c7f-471a-4d95-96a9-dbe73caa475e-c000.csv", "part05.csv")
        shutil.copy("".join(map(str,glob.glob('wiki_cleaned_csv/part*.csv'))), filename+".csv")
        print(filename+".csv created!")
        #time.sleep(5*60)
if __name__ == '__main___':
    sent = "hi this is . preetham , checking  hhhhkh hkhjkhh  hhjhh hkhjkhk hhjhjkh hkkhkh"
    print(list(remove_special_chars(sent)))
