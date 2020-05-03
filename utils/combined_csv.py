"""
@author: Preetham Salehundam
@email: salehundam.2@wright.edu

utility script to combine data files
"""
import pandas as pd


if __name__ == "__main__":
    article_raw_df = pd.read_csv("data/augmented_edits.csv")
    # obtained from running util.py and DataPreprocessing script
    editor_raw_df = pd.read_csv("data/ugment.csv")
    article_df = article_raw_df[["article_id", "articletitle", "editid"]]
    final_df = editor_raw_df.join(article_df, on="editid", how="inner", rsuffix="article")
    final_df = final_df.drop(["editidarticle"], axis=1)
    final_df["editid"] = final_df["editid"].astype("int")
    #final_df["article_id"] = final_df["article_id"].astype("int")
    final_df["editid"] = final_df["editid"].drop_duplicates()
    #final_df["article_id"] = final_df["article_id"].dropna()
    final_df["article_id"] = final_df["article_id"].astype("int")
    final_df = final_df.rename(columns={"article_id": "articleid"})
    print(final_df.info())
    print(final_df.describe())
    final_df.loc[final_df["vulgar_word_ratio"] > 0, "class"] = 1
    final_df["class"] = final_df["class"].astype("category")
    final_df = final_df[~final_df.editid.isin(final_df.articleid)]
    final_df.to_csv("data/final_merged.csv", index=False)

    final_df = final_df.drop(["articleid", "articletitle", "size", "word_count"], axis=1)
    combined_data = pd.read_csv("data/combined-new.csv")
    df = pd.concat([final_df, combined_data], axis=0)
    df.to_csv("data/combined_old.csv", index=False)