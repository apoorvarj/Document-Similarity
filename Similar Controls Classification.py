#!/usr/bin/env python
# coding: utf-8

# In[285]:


# Data Structures
import pandas as pd
import numpy as np
import os
# Utilities
from pprint import pprint
import time

# Visualizations 
import seaborn as sns
import matplotlib.pylab as plt

# Machine Learing 
from sklearn.utils import resample
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import ShuffleSplit

#Natural Language Processing
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
import numpy as np
from tqdm import tqdm

from nltk.corpus import stopwords
from gensim.parsing import preprocessing as pp

stop_words = stopwords.words("english")
stop_words = set(stop_words)


# In[77]:


os.getcwd()


# In[460]:


risks_controls = pd.read_excel("Controls - mapped to top risks -2-27-2020.xlsx")


# In[461]:


risks_controls.shape


# In[462]:


risks_controls.columns.tolist()


# In[278]:


risks_controls


# In[254]:


risks = risks_controls[["Risk ID","Risk Description","New RTC Level 1","New RTC Level 2","Risk Business Group"]]


# In[463]:


controls = risks_controls[["Control ID number","Control Description",'Control Status','Control Effectiveness','Key Control']]


# In[464]:


controls.columns = ["control_id","control_description","control_status","control_effectiveness",'key_control']


# In[180]:


controls


# In[279]:


controls.shape


# In[181]:


risks


# In[182]:


risks.columns = ["control_id","risk_id","risk_description","risk_rtc_level_1","risk_rtc_leverl_2","risk_business_group"]


# In[183]:


risks.shape


# ## Getting control fields 

# In[184]:


diff = pd.read_csv("Diff_Quality_Cluster_Placement_WEX_Ready_Latest.csv")


# In[185]:


diff.shape


# In[186]:


diff


# In[187]:


diff["control_id"] = diff["Control ID"].str.extract('(\d+)')


# In[188]:


diff.columns.tolist()


# In[189]:


diff_controls = diff[["control_id",'Control Name','Control Description','Control Frequency', 
  'Clusters_autolabel_20',
 'Clusters_label_20',
 'Clusters_autolabel_30',
 'Clusters_label_30',
 'Semantic_Cluster',
 'Duplicate Group',
 'DuplicateLabels',
 'Word_Count',
 'Sub_DuplicateLabels',
 'Control Placement',
 'Control Taxonomy',
 'Control Rating',
 'SOX Control Type',
 'SOX Evaluation',
 'Nature of Control (Control Design)',
 'Key Control',
  'Duplicate Control ID',
 'Quality_Rating']]


# In[190]:


diff_controls = diff[["control_id",'Control Name','Control Description','Control Frequency','Control Placement',
                     'Control Taxonomy']]


# In[191]:


diff_controls = diff_controls.drop_duplicates()


# In[192]:


diff_controls.shape


# In[193]:


diff_controls.isnull().sum()


# In[194]:


diff_controls


# In[195]:


diff_controls[diff_controls.duplicated(['control_id'],keep=False)]


# In[196]:


diff_controls.drop_duplicates(["control_id"]).shape


# In[197]:


controls


# In[199]:


#filter for the controls we have
controls.shape


# In[200]:


controls.dtypes


# In[201]:


controls.isnull().sum()


# In[202]:


controls["control_id"]  = controls["control_id"].astype(str)


# In[203]:


controls_1 = controls.merge(diff_controls, how = "left", on = ["control_id","control_id"])


# In[204]:


controls_1.shape


# In[205]:


controls_1.isnull().sum()


# In[206]:


len(controls_1["control_id"].unique())


# In[281]:


controls_bkp =controls


# In[282]:


controls


# In[283]:


controls = controls[["control_id","control_description"]]


# ## Natural Language Processing

# In[286]:


stop_words = stopwords.words("english")
stop_words = set(stop_words)

pp_list = [
    lambda x: x.lower(),
    pp.strip_tags,
    pp.strip_multiple_whitespaces,
    pp.strip_punctuation,
    pp.strip_short
          ]

def tokenizer(line):
    """ Applies the following steps in sequence:
        Converts to lower case,
       Strips tags (HTML and others),
        Strips multiple whitespaces,
        Strips punctuation,
        Strips short words(min lenght = 3),
        --------------------------
        :param line: a document
        
        Returns a list of tokens"""
    
    tokens = pp.preprocess_string(line, filters=pp_list)
    return tokens


# In[287]:


get_ipython().run_cell_magic('time', '', "\ntrain_texts = []\n\nfor line in controls[['control_description']].fillna(' ').values:\n    train_texts.append(tokenizer(line[0]))#+' '+line[1]))")


# In[288]:


get_ipython().run_cell_magic('time', '', '\nimport gensim\nbigram = gensim.models.Phrases(train_texts)\nbigram_phraser = gensim.models.phrases.Phraser(bigram)\ntokens_ = bigram_phraser[train_texts]\ntrigram = gensim.models.Phrases(tokens_)\ntrigram_phraser = gensim.models.phrases.Phraser(trigram)\n\nfrom nltk.stem import WordNetLemmatizer\nlemmatizer = WordNetLemmatizer()\n\nfrom nltk.stem.snowball import SnowballStemmer\nstemmer = SnowballStemmer("english")\n\ndef process_texts(tokens):\n    """Removes stop words, Stemming,\n       Lemmatization assuming verb"""\n    \n    tokens = [token for token in tokens if token not in stop_words]\n    tokens = bigram_phraser[tokens]\n    tokens = trigram_phraser[tokens]\n#     tokens = [stemmer.stem(token) for token in tokens]\n    tokens = [lemmatizer.lemmatize(word, pos=\'v\') for word in tokens]\n    return tokens')


# In[289]:


get_ipython().run_cell_magic('time', '', "\nfinal_texts = []\n\nfor line in train_texts:\n    final_texts.append(process_texts(line))\n\ncontrols['final_tokens'] = final_texts\ncontrols.head()")


# In[290]:


controls['control_input_text'] = controls['final_tokens'].str.join(' ')


# In[291]:


controls


# In[295]:


controls.to_csv("controls_input.csv")


# In[292]:


stop_words = stopwords.words("english")
stop_words = set(stop_words)

pp_list = [
    lambda x: x.lower(),
    pp.strip_tags,
    pp.strip_multiple_whitespaces,
    pp.strip_punctuation,
    pp.strip_short
          ]

def tokenizer(line):
    """ Applies the following steps in sequence:
        Converts to lower case,
       Strips tags (HTML and others),
        Strips multiple whitespaces,
        Strips punctuation,
        Strips short words(min lenght = 3),
        --------------------------
        :param line: a document
        
        Returns a list of tokens"""
    
    tokens = pp.preprocess_string(line, filters=pp_list)
    return tokens


# In[293]:


get_ipython().run_cell_magic('time', '', "\ntrain_texts = []\n\nfor line in risks[['risk_description']].fillna(' ').values:\n    train_texts.append(tokenizer(line[0]))#+' '+line[1]))")


# In[294]:


get_ipython().run_cell_magic('time', '', '\nimport gensim\nbigram = gensim.models.Phrases(train_texts)\nbigram_phraser = gensim.models.phrases.Phraser(bigram)\ntokens_ = bigram_phraser[train_texts]\ntrigram = gensim.models.Phrases(tokens_)\ntrigram_phraser = gensim.models.phrases.Phraser(trigram)\n\nfrom nltk.stem import WordNetLemmatizer\nlemmatizer = WordNetLemmatizer()\n\nfrom nltk.stem.snowball import SnowballStemmer\nstemmer = SnowballStemmer("english")\n\ndef process_texts(tokens):\n    """Removes stop words, Stemming,\n       Lemmatization assuming verb"""\n    \n    tokens = [token for token in tokens if token not in stop_words]\n    tokens = bigram_phraser[tokens]\n    tokens = trigram_phraser[tokens]\n#     tokens = [stemmer.stem(token) for token in tokens]\n    tokens = [lemmatizer.lemmatize(word, pos=\'v\') for word in tokens]\n    return tokens')


# In[96]:


get_ipython().run_cell_magic('time', '', '\nfinal_texts = []\n\nfor line in train_texts:\n    final_texts.append(process_texts(line))')


# In[97]:


risks['risk_final_tokens'] = final_texts
risks


# In[98]:


risks['risk_input_text'] = risks['risk_final_tokens'].str.join(' ')

risks


# In[99]:


risks.shape


# In[100]:


#Exgtract control to risks mapping
mapping = risks[["control_id","risk_id"]]


# In[202]:


risk = risks[["risk_id","risk_description","risk_input_text"]]


# In[203]:


risk.shape


# In[204]:


risk


# In[205]:


risk = risk.drop_duplicates()


# In[206]:


risk.shape


# In[207]:


risk


# In[208]:


risk.reset_index(drop=True,inplace=True)


# In[209]:


risk


# ## Creating Mapping files for controls

# In[210]:


controls_1


# In[220]:


controls_1.columns = ["control_id","control_description","control_status","control_effectiveness","control_name",
                      "control__description","control_frequency","control_placement","control_taxonomy"]


# In[221]:


controls_mapping_1 = controls_1[["control_id","control_description"]]


# In[222]:


controls_mapping_1


# In[223]:


controls_mapping_1 = controls_1[["control_id","control_description"]]
controls_mapping_2 = controls_1[["control_id","control_status"]]
controls_mapping_3 = controls_1[["control_id","control_effectiveness"]]
controls_mapping_4 = controls_1[["control_id","control_frequency"]]
controls_mapping_5 = controls_1[["control_id","control_placement"]]
controls_mapping_6 = controls_1[["control_id","control_taxonomy"]]


# In[224]:


controls_dict_1 = dict(zip(controls_mapping_1.control_id, controls_mapping_1.control_description))
controls_dict_2 = dict(zip(controls_mapping_2.control_id, controls_mapping_2.control_status))
controls_dict_3 = dict(zip(controls_mapping_3.control_id, controls_mapping_3.control_effectiveness))
controls_dict_4 = dict(zip(controls_mapping_4.control_id, controls_mapping_4.control_frequency))
controls_dict_5 = dict(zip(controls_mapping_5.control_id, controls_mapping_5.control_placement))
controls_dict_6 = dict(zip(controls_mapping_6.control_id, controls_mapping_6.control_taxonomy))


# In[225]:


controls_dict_1


# In[466]:


#Getting Key Control
controls.dtypes


# In[468]:


controls["control_id"] = controls["control_id"].astype(str)


# In[469]:


controls_dict_13 = dict(zip(controls.control_id,controls.key_control))


# In[283]:


controls_mapping.to_csv("control_mapping.csv")


# ## Creating tf-idf vectors for controls

# In[296]:


# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 0.8


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        max_features = 3000,
                        norm='l2',
                        sublinear_tf=True)


# In[297]:


from sklearn.decomposition import TruncatedSVD
svd_model = TruncatedSVD(n_components=2000, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)


from sklearn.pipeline import Pipeline
svd_transformer = Pipeline([('tfidf', tfidf), 
                            ('svd', svd_model)])
svd_matrix = svd_transformer.fit_transform(controls["control_input_text"])


# In[298]:


svd_matrix


# In[299]:


pd.DataFrame(svd_matrix)


# In[212]:


explained_variance = svd_model.explained_variance_ratio_.sum()
print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))


# In[213]:


svd_matrix.shape


# In[214]:


from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse


# In[133]:


get_ipython().run_cell_magic('time', '', 'svd_sparse = sparse.csr_matrix(svd_matrix)\nsimilarities_svd_sparse = cosine_similarity(svd_sparse,dense_output = False)')


# In[307]:


print(similarities_svd_sparse)


# In[134]:


control_sim = pd.DataFrame(similarities_svd_sparse.todense())


# In[135]:


control_sim.shape


# In[136]:


control_sim


# In[137]:


control_sim.shape


# In[362]:


get_ipython().run_cell_magic('time', '', 'nlargest = 25\norder = np.argsort(-control_sim.values, axis=1)[:, :nlargest]')


# In[363]:


get_ipython().run_cell_magic('time', '', "control_sim_result = pd.DataFrame(control_sim.columns[order], \n                      columns=['top{}'.format(i) for i in range(1, nlargest+1)],\n                      index=control_sim.index)")


# In[364]:


control_sim_result


# In[436]:


control_sim_result.shape


# In[383]:


control_sim_stacked = pd.DataFrame(control_sim_result.stack())


# In[384]:


control_sim_stacked.reset_index(inplace = True)


# In[385]:


control_sim_stacked.columns  = ["control_id_1","top_n_x","control_id_2"]


# In[386]:


control_sim_stacked


# In[377]:


control_sim_stacked.shape


# In[389]:


controls_mapping


# In[395]:


control_sim_stacked["control_id_1_value"] = control_sim_stacked["control_id_1"].map(controls_dict)


# In[397]:


control_sim_stacked["control_id_2_value"] = control_sim_stacked["control_id_2"].map(controls_dict)


# In[402]:


control_sim_stacked["control_id_1_desc"] = control_sim_stacked["control_id_1"].map(controls_dict_1)


# In[390]:


controls_dict = dict(zip(controls_mapping.index, controls_mapping.control_id))


# In[392]:


len(controls_dict)


# In[404]:


control_sim_stacked["control_id_2_desc"] = control_sim_stacked["control_id_2"].map(controls_dict_1)


# In[635]:


control_sim_stacked["control_id_1_status"] = control_sim_stacked["control_id_1"].map(controls_dict_2)
control_sim_stacked["control_id_2_status"] = control_sim_stacked["control_id_2"].map(controls_dict_2)


# In[636]:


control_sim_stacked["control_id_1_effectiveness"] = control_sim_stacked["control_id_1"].map(controls_dict_3)
control_sim_stacked["control_id_2_effectiveness"] = control_sim_stacked["control_id_2"].map(controls_dict_3)


# In[691]:


controls_dict_6


# In[692]:


control_sim_stacked["control_id_1_placement"] = control_sim_stacked["control_id_1"].map(controls_dict_5)
control_sim_stacked["control_id_2_placement"] = control_sim_stacked["control_id_2"].map(controls_dict_5)


# In[693]:


control_sim_stacked["control_id_1_frequency"] = control_sim_stacked["control_id_1"].map(controls_dict_4)
control_sim_stacked["control_id_2_frequency"] = control_sim_stacked["control_id_2"].map(controls_dict_4)


# In[694]:


control_sim_stacked["control_id_1_taxonomy"] = control_sim_stacked["control_id_1"].map(controls_dict_6)
control_sim_stacked["control_id_2_taxonomy"] = control_sim_stacked["control_id_2"].map(controls_dict_6)


# In[695]:


control_sim_stacked


# In[638]:


control_sim_stacked.shape


# In[407]:


control_sim_values_result = pd.DataFrame([control_sim.values[i,order[i]] for i in range(len(order))], 
                      columns=['top{}'.format(i) for i in range(1, nlargest+1)],
                      index=control_sim.index)


# In[408]:


control_sim_svalues_tacked = pd.DataFrame(control_sim_values_result.stack())


# In[409]:


control_sim_svalues_tacked.shape


# In[410]:


control_sim_svalues_tacked


# In[411]:


control_sim_svalues_tacked.reset_index(inplace = True)


# In[412]:


control_sim_svalues_tacked.columns  = ["control_id","top_n_y","similarity_measure"]


# In[413]:


control_sim_svalues_tacked


# In[696]:


control_similarity_total = pd.concat([control_sim_stacked,control_sim_svalues_tacked],axis=1, sort=False)


# In[697]:


control_similarity_total.shape


# In[698]:


control_similarity_total


# In[699]:


control_similarity_total.drop(columns=["top_n_y","control_id"],axis=1,inplace=True)


# In[700]:


control_similarity_total


# In[701]:


control_similarity_total.to_csv("controls_similarity_0229.csv")


# ## Finding risk similarity 

# ## Creating Mapping files for risks

# In[255]:


risks


# In[256]:


risks = risks.drop_duplicates()


# In[257]:


risks.shape


# In[262]:


risks


# In[261]:


risks.columns = ["risk_id","risk_desc","risk_rtc_1","risk_rtc_2","risk_business_group"]


# In[263]:


risks_mapping_1 = risks[["risk_id","risk_desc"]]
risks_mapping_2 = risks[["risk_id","risk_rtc_1"]]
risks_mapping_3 = risks[["risk_id","risk_rtc_2"]]
risks_mapping_4 = risks[["risk_id","risk_business_group"]]


# In[266]:


risks_dict_1 = dict(zip(risks.risk_id, risks.risk_desc))
risks_dict_2 = dict(zip(risks.risk_id, risks.risk_rtc_1))
risks_dict_3 = dict(zip(risks.risk_id, risks.risk_rtc_2))
risks_dict_4 = dict(zip(risks.risk_id, risks.risk_business_group))


# ## TF-IDF on risks

# In[251]:


# Parameter election
ngram_range = (1,2)
min_df = 10
max_df = 0.8


tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=ngram_range,
                        stop_words=None,
                        lowercase=False,
                        max_df=max_df,
                        min_df=min_df,
                        norm='l2',
                        sublinear_tf=True)


# In[220]:


svd_model = TruncatedSVD(n_components=1500, 
                         algorithm='randomized',
                         n_iter=10, random_state=42)


from sklearn.pipeline import Pipeline
svd_transformer = Pipeline([('tfidf', tfidf), 
                            ('svd', svd_model)])
svd_matrix = svd_transformer.fit_transform(risk["risk_input_text"])


# In[221]:


explained_variance = svd_model.explained_variance_ratio_.sum()
print("Sum of explained variance ratio: %d%%" % (int(explained_variance * 100)))


# In[428]:


svd_matrix.shape


# In[223]:


get_ipython().run_cell_magic('time', '', 'risks_svd_sparse = sparse.csr_matrix(svd_matrix)\nrisks_similarities_svd_sparse = cosine_similarity(risks_svd_sparse,dense_output = False)')


# In[429]:


print(risks_similarities_svd_sparse)


# In[252]:


risks_sim = pd.DataFrame(risks_similarities_svd_sparse.todense())


# In[430]:


risks_sim.shape


# In[431]:


risks_sim


# In[257]:


risks_sim.shape


# In[432]:


get_ipython().run_cell_magic('time', '', 'nlargest = 25\norder = np.argsort(-risks_sim.values, axis=1)[:, :nlargest]')


# In[433]:


get_ipython().run_cell_magic('time', '', "risk_sim_result = pd.DataFrame(risks_sim.columns[order], \n                      columns=['top{}'.format(i) for i in range(1, nlargest+1)],\n                      index=risks_sim.index)")


# In[434]:


risk_sim_result.shape


# In[435]:


risk_sim_result


# In[440]:


risk_sim_stacked = pd.DataFrame(risk_sim_result.stack())


# In[441]:


risk_sim_stacked


# In[442]:


risk_sim_stacked.reset_index(inplace = True)


# In[445]:


risk_sim_stacked


# In[444]:


risk_sim_stacked.columns  = ["risk_id_1","top_n_x","risk_id_2"]


# In[446]:


risk_sim_stacked.shape


# In[485]:


risks_dict


# In[449]:


risk_sim_stacked["risk_id_1_value"] = risk_sim_stacked["risk_id_1"].map(risks_dict)

risk_sim_stacked["risk_id_2_value"] = risk_sim_stacked["risk_id_2"].map(risks_dict)

risk_sim_stacked["risk_id_1_desc"] = risk_sim_stacked["risk_id_1"].map(risks_dict_1)


# In[451]:


risk_sim_stacked["risk_id_2_desc"] = risk_sim_stacked["risk_id_2"].map(risks_dict_1)


# In[452]:


risk_sim_stacked


# In[453]:


risk_sim_values_result = pd.DataFrame([risks_sim.values[i,order[i]] for i in range(len(order))], 
                      columns=['top{}'.format(i) for i in range(1, nlargest+1)],
                      index=risks_sim.index)


# In[455]:


risk_sim_values_result.shape


# In[456]:


risk_sim_values_result


# In[457]:


risk_sim_values_stacked = pd.DataFrame(risk_sim_values_result.stack())


# In[458]:


risk_sim_values_stacked


# In[459]:


risk_sim_values_stacked.shape


# In[460]:


risk_sim_values_stacked.reset_index(inplace = True)


# In[461]:


risk_sim_values_stacked.columns  = ["risk_id","top_n_y","similarity_measure"]


# In[462]:


risk_sim_values_stacked


# In[463]:


risk_similarity_total = pd.concat([risk_sim_stacked,risk_sim_values_stacked],axis=1, sort=False)


# In[464]:


risk_similarity_total.shape


# In[466]:


risk_similarity_total


# In[467]:


risk_similarity_total.drop(columns=["top_n_y","risk_id"],axis=1,inplace=True)


# In[506]:


risk_similarity_total


# In[497]:


risks_dict_3


# In[500]:


risk_similarity_total.drop(inplace = True, columns= "risk_id_1_ rtc_2")


# In[504]:


risk_similarity_total["risk_id_1_rtc_1"] = risk_similarity_total["risk_id_1_value"].map(risks_dict_1)
risk_similarity_total["risk_id_1_rtc_2"] = risk_similarity_total["risk_id_1_value"].map(risks_dict_2)
risk_similarity_total["risk_id_1_business_group"] = risk_similarity_total["risk_id_1_value"].map(risks_dict_3)


# In[505]:


risk_similarity_total["risk_id_2_rtc_1"] = risk_similarity_total["risk_id_2_value"].map(risks_dict_1)
risk_similarity_total["risk_id_2_rtc_2"] = risk_similarity_total["risk_id_2_value"].map(risks_dict_2)
risk_similarity_total["risk_id_2_business_group"] = risk_similarity_total["risk_id_2_value"].map(risks_dict_3)


# In[703]:


risk_similarity_total.to_csv("risks_similarity_0228_1.csv")


# In[702]:


risk_similarity_total


# In[ ]:





# ## Joining the risk and controls together

# In[5]:


import pandas as pd
risks = pd.read_csv("risks_similarity_0228_1.csv")
controls = pd.read_csv("controls_similarity_0229.csv")


# In[6]:


risks.shape


# In[7]:


risks


# In[8]:


risks.isnull().sum()


# In[9]:


risks = risks.dropna()


# In[10]:


risks.isnull().sum()


# In[11]:


controls.shape


# In[12]:


controls.isnull().sum()


# In[13]:


controls


# In[14]:


risks_controls = pd.read_excel("Controls - mapped to top risks -2-27-2020.xlsx")


# In[15]:


risks_controls.shape


# In[16]:


risks_controls.columns.tolist()


# In[87]:


risks_controls_1 = risks_controls[["Control ID number","Risk ID"]]


# In[88]:


risks_controls_1


# In[89]:


risks_controls_1.dtypes


# In[90]:


risks_controls_1.isnull().sum()


# In[91]:


risks_controls_1.shape


# In[92]:


risks_controls_1


# In[35]:


risks.shape


# In[53]:


risks['similarity_measure_round']  = risks['similarity_measure'].round(decimals=3)


# In[54]:


risks


# In[55]:


risks_1 =  risks.loc[(risks['similarity_measure_round'] < 1.000)]


# In[56]:


print(risks.shape)
print(risks_1.shape)


# In[58]:


risks_1 =  risks_1.loc[(risks_1['similarity_measure_round'] > 0.250)]
print(risks_1.shape)


# In[60]:


risks_1.describe()


# In[59]:


risks_1


# In[75]:


#Taking only active controls
controls_1 = controls.loc[(controls['control_id_1_status'] == "Active") & (controls['control_id_2_status'] == "Active")]


# In[76]:


print(controls.shape)
print(controls_1.shape)


# In[77]:


controls_1.describe()


# In[78]:


controls_1['similarity_measure_round']  = controls_1['similarity_measure'].round(decimals=3)
controls_2 =  controls_1.loc[(controls_1['similarity_measure_round'] < 1.000)]


# In[79]:


controls_2.shape


# In[80]:


controls_3 =  controls_2.loc[(controls_2['similarity_measure_round'] >= 0.250)]


# In[81]:


controls_3.shape


# In[32]:


risks_1


# In[151]:


risks_1.to_csv("risks_similarity_0302.csv",index = False)


# In[99]:


controls_3


# In[160]:


controls_3.shape


# In[93]:


risks_controls_1


# In[94]:


risks_controls_1.columns = ["control_id","risk_id"]


# In[95]:


risks_controls_dict = dict(zip(risks_controls_1.control_id, risks_controls_1.risk_id))


# In[96]:


risks_controls_dict


# In[100]:


controls_3["control_id_1_risk_id"] = controls_3["control_id_1_value"].map(risks_controls_dict)


# In[153]:


controls_3.to_csv("controls_similarity_0302.csv",index=False)


# In[152]:


controls_3.shape


# In[114]:


controls_3[["control_id_1_value","control_id_1_risk_id"]]


# In[104]:


controls_3.columns.tolist()


# In[115]:


controls_4= controls_3.groupby('control_id_1_value').apply(lambda x: [list(x['control_id_2_value']), 
                                                                        list(x['control_id_2_desc']), 
                                                                        list(x['control_id_2_effectiveness']), 
                                                                        list(x['control_id_2_placement']), 
                                                                        list(x['control_id_2_frequency']),
                                                                        list(x['control_id_2_taxonomy']),
                                                                     list(x['similarity_measure_round'])]).apply(pd.Series)


# In[116]:


controls_4.columns = ['list_of_control_id_2_value',
                     'list_of_control_id_2_desc',
                     'list_of_control_id_2_effectiveness',
                    'list_of_control_id_2_placement',
                    'list_of_control_id_2_frequency',
                    'list_of_control_id_2_taxonomy',
                    'list_of_similarity_measure_round']


# In[119]:


controls_4


# In[118]:


controls_4.reset_index(inplace=True)


# In[120]:


controls_4["control_id_1_risk_id"] = controls_4["control_id_1_value"].map(risks_controls_dict)


# In[121]:


controls_4


# In[122]:


risks_controls_dict


# In[123]:


controls_4.shape


# In[124]:


controls_4.to_csv("risks_controls_similarity_0301.csv")


# In[167]:


controls_4.columns.tolist()


# In[168]:


controls_7 = controls_4[['control_id_1_risk_id','control_id_1_value',
                        'list_of_control_id_2_value',
                         'list_of_control_id_2_desc',
                         'list_of_control_id_2_effectiveness','list_of_control_id_2_placement',
                         'list_of_control_id_2_frequency','list_of_control_id_2_taxonomy',
                         'list_of_similarity_measure_round']]


# In[169]:


controls_7


# In[231]:


controls_7['control_id_1_risk_id'] = controls_7['control_id_1_risk_id'].astype(str)
controls_7['control_id_1_value'] = controls_7['control_id_1_value'].astype(str)


# In[232]:


controls_7.dtypes


# In[238]:


controls_dict_6


# In[239]:


controls_7["control_id_1_desc"] = controls_7["control_id_1_value"].map(controls_dict_1)
controls_7["control_id_1_status"] = controls_7["control_id_1_value"].map(controls_dict_2)
controls_7["control_id_1_effectiveness"] = controls_7["control_id_1_value"].map(controls_dict_3)
controls_7["control_id_1_freqeuncy"] = controls_7["control_id_1_value"].map(controls_dict_4)
controls_7["control_id_1_placement"] = controls_7["control_id_1_value"].map(controls_dict_5)
controls_7["control_id_1_taxonomy"] = controls_7["control_id_1_value"].map(controls_dict_6)


# In[240]:


controls_7


# In[241]:


controls_7.isnull().sum()


# In[273]:


controls_7['control_id_1_risk_id'] = controls_7['control_id_1_risk_id'].astype(int)


# In[269]:


risks_dict_3


# In[274]:


controls_7["risk_desc"] = controls_7["control_id_1_risk_id"].map(risks_dict_1)
controls_7["risk_rtc_1"] = controls_7["control_id_1_risk_id"].map(risks_dict_2)
controls_7["risk_rtc_2"] = controls_7["control_id_1_risk_id"].map(risks_dict_3)
controls_7["risk_business_group"] = controls_7["control_id_1_risk_id"].map(risks_dict_4)


# In[275]:


controls_7


# In[301]:


controls_7.columns = ["Risk_ID","Control ID","List of Similar Controls","List of Similar Controls Descriptions",
                                 "List of Similar Controls Effectiveness","List of Similar Controls Placement","List of Similar Controls Frequency",
                                 "List of Similar Controls Taxonomy","Similarity Measure","Control Description","Control Status",
                                  "Control Effectiveness","Control Frequency","Control Placement","Control Taxonomy","Risk Description","Risk Taxonomy Level 1",
                                  "Risk Taxonomy Level 2","Risk Business Group"]


# In[302]:


controls_7.columns.tolist()


# In[304]:


controls_7 = controls_7[['Risk ID','Risk Description','Risk Taxonomy Level 1','Risk Taxonomy Level 2','Risk Business Group',
                        'Control ID','Control Description','Control Status','Control Effectiveness','Control Frequency',
                         'Control Placement','Control Taxonomy', 'List of Similar Controls','List of Similar Controls Descriptions',
 'List of Similar Controls Effectiveness',
 'List of Similar Controls Placement',
 'List of Similar Controls Frequency',
 'List of Similar Controls Taxonomy','Similarity Measure']]


# In[305]:


controls_7


# In[308]:


controls_7.isnull().sum()


# In[388]:


controls_8 = controls_7


# In[389]:


controls_8 = controls_8.fillna("Blank")


# In[390]:


controls_8.isnull().sum()


# In[313]:


# Grouping by Risk to find control information
risk_group_1 = controls_8[["Risk_ID","Control ID"]]
risk_group_2 = controls_8[["Risk_ID","Control ID","Control Effectiveness"]]
risk_group_3 = controls_8[["Risk_ID","Control ID","Control Placement"]]
risk_group_4 = controls_8[["Risk_ID","Control ID","Control Frequency"]]


# In[326]:


risk_group_1_gp = risk_group_1.groupby('Risk_ID').count()


# In[327]:


risk_group_1_gp.shape


# In[331]:


risk_group_2_gp = risk_group_2.groupby(['Risk_ID','Control Effectiveness']).count().reset_index()


# In[332]:


risk_group_2_gp


# In[334]:


risk_group_2_gp_up = risk_group_2_gp.pivot(index='Risk_ID', columns='Control Effectiveness')['Control ID']
risk_group_2_gp_up = risk_group_2_gp_up.fillna(0.0)


# In[337]:


risk_group_2_gp_up


# In[338]:


risk_group_3_gp = risk_group_3.groupby(['Risk_ID','Control Placement']).count().reset_index()
risk_group_3_gp_up = risk_group_3_gp.pivot(index='Risk_ID', columns='Control Placement')['Control ID']
risk_group_3_gp_up = risk_group_3_gp_up.fillna(0.0)


# In[339]:


risk_group_3_gp_up


# In[340]:


print(risk_group_3_gp_up.shape)
print(risk_group_2_gp_up.shape)
print(risk_group_1_gp.shape)


# In[344]:


risk_group_1_gp = risk_group_1_gp.reset_index()


# In[348]:


risk_group_12 = risk_group_1_gp.merge(risk_group_2_gp_up,how = "outer",left_on = "Risk_ID",right_on = "Risk_ID")


# In[350]:


print(risk_group_12.shape)
risk_group_12.isnull().sum()


# In[351]:


risk_group_123 = risk_group_12.merge(risk_group_3_gp_up,how = "outer",left_on = "Risk_ID",right_on = "Risk_ID")


# In[359]:


risk_group_12


# In[352]:


print(risk_group_123.shape)
risk_group_123.isnull().sum()


# In[354]:


risk_group_123.to_csv("Risk_Groupby_Control_Agg.csv",index = False)


# In[355]:


risk_group_123


# In[356]:


risk_group_123.dtypes


# In[358]:


risk_group_123


# In[367]:


risk_group_123.columns = ["Risk_ID","Control_ID","Effectiveness_Blanks","Effective","Not_Effective","Partially_Effective",
                          "Placement_Blanks","Detective",
                          "Preventive"]


# In[368]:


risk_group_123


# In[371]:


risk_mapping_5 = risk_group_123[["Risk_ID","Control_ID"]]
risk_mapping_6 = risk_group_123[["Risk_ID","Effectiveness_Blanks"]]
risk_mapping_7 = risk_group_123[["Risk_ID","Effective"]]
risk_mapping_8 = risk_group_123[["Risk_ID","Not_Effective"]]
risk_mapping_9 = risk_group_123[["Risk_ID","Partially_Effective"]]
risk_mapping_10 = risk_group_123[["Risk_ID","Placement_Blanks"]]
risk_mapping_11 = risk_group_123[["Risk_ID","Detective"]]
risk_mapping_12 = risk_group_123[["Risk_ID","Preventive"]]


# In[372]:


risks_dict_5 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Control_ID ))
risks_dict_6 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Effectiveness_Blanks))
risks_dict_7 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Effective))
risks_dict_8 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Not_Effective))
risks_dict_9 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Partially_Effective))
risks_dict_10 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Placement_Blanks))
risks_dict_11 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Detective))
risks_dict_12 = dict(zip(risk_group_123.Risk_ID, risk_group_123.Preventive))


# In[391]:


controls_8


# In[393]:


controls_8["Count of Control IDs"] = controls_8["Risk_ID"].map(risks_dict_5)
controls_8["Count of Blanks (Effectiveness Rating)"] = controls_8["Risk_ID"].map(risks_dict_6)
controls_8["Count of Effective Controls (Effectiveness Rating)"] = controls_8["Risk_ID"].map(risks_dict_7)
controls_8["Count of Not Effective Controls(Effectiveness Rating)"] = controls_8["Risk_ID"].map(risks_dict_8)
controls_8["Count of Partially Effective Controls (Effectiveness Rating)"] = controls_8["Risk_ID"].map(risks_dict_9)
controls_8["Count of Blanks Controls (Control Placement)"] = controls_8["Risk_ID"].map(risks_dict_10)
controls_8["Count of DetectiveControls (Control Placement)"] = controls_8["Risk_ID"].map(risks_dict_11)
controls_8["Count of Preventive Controls(Control Placement)"] = controls_8["Risk_ID"].map(risks_dict_12)


# In[394]:


controls_8


# In[395]:


controls_8.columns.tolist()


# In[397]:


controls_8 = controls_8[['Risk_ID',
 'Risk Description',
 'Risk Taxonomy Level 1',
 'Risk Taxonomy Level 2',
 'Risk Business Group',
 'Count of Control IDs',
 'Count of Blanks (Effectiveness Rating)',
 'Count of Effective Controls (Effectiveness Rating)',
 'Count of Not Effective Controls(Effectiveness Rating)',
 'Count of Partially Effective Controls (Effectiveness Rating)',
 'Count of Blanks Controls (Control Placement)',
 'Count of DetectiveControls (Control Placement)',
 'Count of Preventive Controls(Control Placement)',
 'Control ID',
 'Control Description',
 'Control Status',
 'Control Effectiveness',
 'Control Frequency',
 'Control Placement',
 'Control Taxonomy',
 'List of Similar Controls',
 'List of Similar Controls Descriptions',
 'List of Similar Controls Effectiveness',
 'List of Similar Controls Placement',
 'List of Similar Controls Frequency',
 'List of Similar Controls Taxonomy',
 'Similarity Measure']]


# In[398]:


controls_8


# In[399]:


controls_8.shape


# In[403]:


controls_8["Risk_ID"].nunique()


# In[404]:


controls_8["Control ID"].nunique()


# In[400]:


controls_8.to_csv("risk_control_updated_0303.csv",index = False)


# ## Enhancements post 1 feedback

# In[442]:


#Get the overall control effectiveness rating
rr=pd.read_csv("Risk Register.csv")


# In[443]:


rr.columns.tolist()


# In[409]:


rr = rr[["Risk ID","Overall Control Effectiveness (RCSA)"]]


# In[410]:


rr.shape


# In[411]:


rr.isnull().sum()


# In[420]:


rr.dtypes


# In[419]:


rr


# In[418]:


rr["Risk ID"] =  rr["Risk ID"].str.extract('(\d+)')


# In[421]:


rr["Risk ID"] =  rr["Risk ID"].astype(int)


# In[422]:


rr  = rr.drop_duplicates()


# In[426]:


rr


# In[427]:


rr.columns  = ["risk_id","overall_effectiveness"]


# In[428]:


rr_dict_1 = dict(zip(rr.risk_id, rr.overall_effectiveness))


# In[429]:


rr_dict_1


# In[430]:


controls_8["Overall Effectiveness"] = controls_8["Risk_ID"].map(rr_dict_1)


# In[432]:


controls_8.isnull().sum()


# In[446]:


#Get the inherent risk column
rr_1  = rr[["Risk ID",'Inherent Risk (RCSA)']]


# In[449]:


rr_1.columns = ['risk_id','inherent_risk']


# In[448]:


rr_1.isnull().sum()


# In[453]:


rr_1.dtypes


# In[455]:


rr_1["risk_id"] =  rr_1["risk_id"].str.extract('(\d+)')
rr_1['risk_id'] = rr_1['risk_id'].astype(int)


# In[456]:


rr_dict_2 = dict(zip(rr_1.risk_id,rr_1.inherent_risk))


# In[457]:


controls_8["Inherent Risk"] = controls_8["Risk_ID"].map(rr_dict_2)


# In[458]:


controls_8.isnull().sum()


# In[471]:


#Getting key control
controls_dict_13


# In[472]:


controls_8["Key Control"]  =  controls_8["Control ID"].map(controls_dict_13)


# In[473]:


controls_8.isnull().sum()


# In[474]:


controls_8.shape


# In[476]:


controls_8.to_csv("risk_control_updated_0304.csv")


# In[569]:


controls_8 = pd.read_csv("risk_control_updated_0304.csv")


# In[570]:


controls_8.shape


# In[571]:


controls_8.drop(controls_8.columns[0], axis=1,inplace = True)


# In[572]:


controls_8


# In[573]:


controls_8.columns.tolist()


# In[574]:


controls_8 = controls_8[['Risk_ID',
 'Risk Description',
 'Risk Taxonomy Level 1',
 'Risk Taxonomy Level 2',
 'Risk Business Group',
 'Overall Effectiveness',
 'Inherent Risk',
 'Count of Control IDs',
 'Count of Blanks (Effectiveness Rating)',
 'Count of Effective Controls (Effectiveness Rating)',
 'Count of Not Effective Controls(Effectiveness Rating)',
 'Count of Partially Effective Controls (Effectiveness Rating)',
 'Count of Blanks Controls (Control Placement)',
 'Count of DetectiveControls (Control Placement)',
 'Count of Preventive Controls(Control Placement)',
 'Control ID',
 'Control Description',
 'Control Status',
 'Control Effectiveness',
 'Control Frequency',
 'Control Placement',
 'Key Control',
 'Control Taxonomy',
 'List of Similar Controls',
 'List of Similar Controls Descriptions',
 'List of Similar Controls Effectiveness',
 'List of Similar Controls Placement',
 'List of Similar Controls Frequency',
 'List of Similar Controls Taxonomy',
 'Similarity Measure'   
]]


# In[575]:


#Bring in the high prioirity for review column 
wex = pd.read_csv("Wex_Ready_Distance_First_Quality_Feedback.csv")


# In[576]:


wex.columns.tolist()


# In[577]:


wex_2 = wex[['Control ID', 'Quality_Rating','Quality_Feedback']]


# In[578]:


wex_2["Control ID"] = wex_2["Control ID"].str.extract('(\d+)')


# In[579]:


wex_2.dtypes


# In[580]:


wex_2.columns = ["control_id","Quality_Rating","Quality_Feedback"]


# In[581]:


wex_2["control_id"] = wex_2["control_id"].astype(int)


# In[582]:


wex_dict_1 = dict(zip(wex_2.control_id,wex_2.Quality_Rating))
wex_dict_2 = dict(zip(wex_2.control_id,wex_2.Quality_Feedback))


# In[583]:


controls_8["Control Quality Rating"] = controls_8["Control ID"].map(wex_dict_1)
controls_8["Control Feedback Rating"] = controls_8["Control ID"].map(wex_dict_2)


# In[584]:


controls_8.dtypes


# In[585]:


controls_8.isnull().sum()


# In[539]:


controls_8


# In[586]:


controls_8 = controls_8[['Risk_ID',
 'Risk Description',
 'Risk Taxonomy Level 1',
 'Risk Taxonomy Level 2',
 'Risk Business Group',
 'Overall Effectiveness',
 'Inherent Risk',
 'Count of Control IDs',
 'Count of Blanks (Effectiveness Rating)',
 'Count of Effective Controls (Effectiveness Rating)',
 'Count of Not Effective Controls(Effectiveness Rating)',
 'Count of Partially Effective Controls (Effectiveness Rating)',
 'Count of Blanks Controls (Control Placement)',
 'Count of DetectiveControls (Control Placement)',
 'Count of Preventive Controls(Control Placement)',
 'Control ID',
 'Control Description',
 'Control Status',
 'Control Effectiveness',
 'Control Frequency',
 'Control Placement',
 'Key Control',
 'Control Quality Rating',
 'Control Feedback Rating',                      
 'Control Taxonomy',
 'List of Similar Controls',
 'List of Similar Controls Descriptions',
 'List of Similar Controls Effectiveness',
 'List of Similar Controls Placement',
 'List of Similar Controls Frequency',
 'List of Similar Controls Taxonomy',
 'Similarity Measure'   
]]


# In[587]:


controls_8


# In[588]:


controls_8.to_csv("risk_control_updated_0304_1.csv",index = False)


# In[624]:


controls_8.reset_index(inplace = True)


# In[593]:


controls_8.drop(columns="index",inplace =True)


# In[625]:


controls_8


# In[626]:


controls_8.to_csv("/Users/apoorvarajeshjoshi/Documents/risks.csv",index = False)


# In[628]:


controls_9 = pd.read_csv("/Users/apoorvarajeshjoshi/Documents/risks.csv")


# In[629]:


controls_9


# In[631]:


from ast import literal_eval


# In[632]:


controls_9['List of Similar Controls list'] = controls_9['List of Similar Controls'].apply(lambda x: literal_eval(str(x)))
controls_9['List of Similar Controls Descriptions list'] = controls_9['List of Similar Controls Descriptions'].apply(lambda x: literal_eval(str(x)))


# In[633]:


controls_9['List of Similar Controls list'] = controls_9['List of Similar Controls'].apply(lambda x: literal_eval(str(x)))
controls_9['List of Similar Controls Descriptions list'] = controls_9['List of Similar Controls Descriptions'].apply(lambda x: literal_eval(str(x)))
controls_9['List of Similar Controls Effectiveness list'] = controls_9['List of Similar Controls Effectiveness'].apply(lambda x: literal_eval(str(x)))
controls_9['List of Similar Controls Frequency list'] = controls_9['List of Similar Controls Frequency'].apply(lambda x: literal_eval(str(x)))
controls_9['List of Similar Controls Placement list'] = controls_9['List of Similar Controls Placement'].apply(lambda x: literal_eval(str(x)))
controls_9['List of Similar Controls Taxonomy list'] = controls_9['List of Similar Controls Taxonomy'].apply(lambda x: literal_eval(str(x)))


# In[634]:


controls_9


# In[635]:


print (type(controls_9.at[0, 'List of Similar Controls list']))
print (type(controls_9.at[0, 'List of Similar Controls Descriptions list']))
print (type(controls_9.at[0, 'List of Similar Controls Effectiveness list']))


# In[638]:


controls_9.to_csv("risks_to_be_exploded.csv",index = False)


# In[595]:


controls_8.set_index(['Risk_ID',
 'Risk Description',
 'Risk Taxonomy Level 1',
 'Risk Taxonomy Level 2',
 'Risk Business Group',
 'Overall Effectiveness',
 'Inherent Risk',
 'Count of Control IDs',
 'Count of Blanks (Effectiveness Rating)',
 'Count of Effective Controls (Effectiveness Rating)',
 'Count of Not Effective Controls(Effectiveness Rating)',
 'Count of Partially Effective Controls (Effectiveness Rating)',
 'Count of Blanks Controls (Control Placement)',
 'Count of DetectiveControls (Control Placement)',
 'Count of Preventive Controls(Control Placement)',
 'Control ID',
 'Control Description',
 'Control Status',
 'Control Effectiveness',
 'Control Frequency',
 'Control Placement',
 'Key Control',
 'Control Quality Rating',
 'Control Feedback Rating',                      
 'Control Taxonomy'], drop=True,inplace = True)


# In[596]:


controls_8


# In[597]:


controls_8.shape


# In[616]:





# In[607]:


from platform import python_version

print(python_version())


# In[613]:


import pandas as pd


# In[614]:


pd.__version__


# In[612]:


get_ipython().system('pip install --upgrade pandas==1.0.0rc0')


# In[620]:


controls_9 = pd.read_csv("controls_9.csv")


# In[621]:


controls_9.shape


# In[622]:


controls_9


# In[623]:


controls_9.dtypes


# In[ ]:





# In[ ]:





# In[ ]:





# In[493]:


controls_8_1['List of Similar Controls len'] = controls_8_1["List of Similar Controls"].str.len()
controls_8_1['List of Similar Controls Descriptions len'] = controls_8_1["List of Similar Controls Descriptions"].str.len()
controls_8_1['List of Similar Controls Effectiveness len'] = controls_8_1["List of Similar Controls Effectiveness"].str.len()
controls_8_1['List of Similar Controls Placement len'] = controls_8_1["List of Similar Controls Placement"].str.len()
controls_8_1['List of Similar Controls Frequency len'] = controls_8_1["List of Similar Controls Frequency"].str.len()
controls_8_1['List of Similar Controls Taxonomy len'] = controls_8_1["List of Similar Controls Taxonomy"].str.len()


# In[494]:


controls_8_1


# In[495]:


controls_8_1.columns.tolist()


# In[505]:


controls_8 = controls_8_bkp


# In[506]:


controls_8.shape


# In[509]:


controls_8


# In[508]:


controls_8.isnull().sum()


# In[507]:


controls_8


# In[478]:


lst_col = 'List of Similar Controls'


# In[ ]:


df.set_index(['A']).apply(pd.Series.explode).reset_index()


# In[ ]:





# In[ ]:





# In[496]:


df = controls_8


# In[497]:


controls_9 =pd.DataFrame({
     col:np.repeat(df[col].values, df[lst_col].str.len())
     for col in df.columns.difference([lst_col])
 }).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns.tolist()]


# In[498]:


controls_9.shape


# In[499]:


controls_9


# In[502]:


controls_9.to_csv("to_be_checked.csv")


# In[487]:


df = controls_9
lst_col = 'List of Similar Controls Descriptions'
controls_10 =pd.DataFrame({
     col:np.repeat(df[col].values, df[lst_col].str.len())
     for col in df.columns.difference([lst_col])
 }).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns.tolist()]


# In[488]:


controls_10.shape


# In[486]:


controls_8.shape


# In[ ]:





# In[ ]:


df = controls_10
lst_col = 'List of Similar Controls Effectiveness'
controls_10 =pd.DataFrame({
     col:np.repeat(df[col].values, df[lst_col].str.len())
     for col in df.columns.difference([lst_col])
 }).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns.tolist()]


# In[ ]:


df = controls_11
lst_col = 'List of Similar Controls Frequency'
controls_10 =pd.DataFrame({
     col:np.repeat(df[col].values, df[lst_col].str.len())
     for col in df.columns.difference([lst_col])
 }).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns.tolist()]


# In[ ]:


df = controls_11
lst_col = 'List of Similar Controls Taxonomy'
controls_10 =pd.DataFrame({
     col:np.repeat(df[col].values, df[lst_col].str.len())
     for col in df.columns.difference([lst_col])
 }).assign(**{lst_col:np.concatenate(df[lst_col].values)})[df.columns.tolist()]


# In[276]:


controls_7.to_csv("risk_control_updated_0302.csv")


# In[125]:


risks_1


# In[128]:


risks_2= risks_1.groupby('risk_id_1_value').apply(lambda x: [list(x['risk_id_2_value']), 
                                                                        list(x['risk_id_2_desc']), 
                                                                        list(x['risk_id_2_rtc_2']), 
                                                                        list(x['risk_id_2_rtc_1']), 
                                                                        list(x['risk_id_2_business_group']),
                                                                        list(x['similarity_measure_round'])]).apply(pd.Series)


# In[129]:


risks_2.columns = ['list_of_risk_id_2_value',
                     'list_of_risk_id_2_desc',
                     'list_of_risk_id_2_rtc_2',
                    'list_of_risk_id_2_rtc_1',
                    'list_of_risk_id_2_business_group',
                    'list_of_similarity_measure_round']


# In[130]:


risks_2


# In[133]:


risks_2.reset_index(inplace = True)


# In[134]:


risks_2.shape


# In[138]:


dict_1 = dict(zip(risks_2.risk_id_1_value, risks_2.list_of_risk_id_2_value))
dict_2 = dict(zip(risks_2.risk_id_1_value, risks_2.list_of_risk_id_2_desc))
dict_3 = dict(zip(risks_2.risk_id_1_value, risks_2.list_of_risk_id_2_rtc_2))
dict_4 = dict(zip(risks_2.risk_id_1_value, risks_2.list_of_risk_id_2_rtc_1))
dict_5 = dict(zip(risks_2.risk_id_1_value, risks_2.list_of_risk_id_2_business_group))
dict_6 = dict(zip(risks_2.risk_id_1_value, risks_2.list_of_similarity_measure_round))


# In[139]:


controls_4["list_of_similar_risks_value"] = controls_4["control_id_1_risk_id"].map(dict_1)
controls_4["list_of_similar_risks_desc"] = controls_4["control_id_1_risk_id"].map(dict_2)
controls_4["list_of_similar_risks_rt2"] = controls_4["control_id_1_risk_id"].map(dict_3)
controls_4["list_of_similar_risks_rt1"] = controls_4["control_id_1_risk_id"].map(dict_4)
controls_4["list_of_similar_risks_business_groups"] = controls_4["control_id_1_risk_id"].map(dict_5)
controls_4["list_of_similar_risks_similarity_measure"] = controls_4["control_id_1_risk_id"].map(dict_6)


# In[126]:


controls_4


# In[127]:


risks_controls_dict


# In[140]:


controls_4


# In[141]:


controls_4.shape


# In[146]:


controls_5 = controls_4[["control_id_1_risk_id","list_of_similar_risks_value",
                         "list_of_similar_risks_desc",
                         "list_of_similar_risks_rt1",
                         "list_of_similar_risks_business_groups",
                         "list_of_similar_risks_similarity_measure",
                         "control_id_1_value","list_of_control_id_2_value",
                         "list_of_control_id_2_desc","list_of_control_id_2_effectiveness",
                         "list_of_control_id_2_placement", 
                         "list_of_control_id_2_frequency",
                         "list_of_control_id_2_taxonomy","list_of_similarity_measure_round"]]


# In[147]:


controls_5


# In[148]:


controls_5.columns = ["risk_id","list_of_similar_risks","list_of_similar_risks_desc",
                     "list_of_similar_risks_rtc1","list_of_similar_risks_business_groups",
                      "list_of_similar_risks_similarity_measure","corressponding_control_id","list_of_similar_controls",
                      "list_of_similar_controls_desc","list_of_similar_controls_effectiveness","list_of_control_id_2_placement",
                      "list_of_control_id_2_frequency","list_of_control_id_2_taxonomy","list_of_similarity_measure"]
                       


# In[150]:


controls_5


# In[149]:


controls_5.to_csv("risk_control_similarity_full_0302.csv")


# In[142]:


controls_4.to_csv("risk_control_similarity_full_0301.csv")


# In[ ]:




