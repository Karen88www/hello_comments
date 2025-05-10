# use bert_venv
# pip install bertopic

import pandas as pd
from bertopic import BERTopic

# 測試comment_df_1
documents = pd.read_csv('comments_data/comments/df_1.csv')
print(">> cat df_1:", documents.head(5))
exit()

topic_model = BERTopic()
topics, _ = topic_model.fit_transform(documents['comment_text'])