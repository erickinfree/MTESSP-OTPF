from transformers import BertTokenizer
from model import BertForMultiLabelClassification
from multilabel_pipeline import MultiLabelPipeline
from pprint import pprint
import pandas as pd 
import os 
from tqdm import tqdm

# res_list = [['neutral'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['annoyance', 'confusion'], ['approval'], ['desire'], ['neutral'], ['disappointment'], ['approval', 'neutral'], ['neutral'], ['disappointment'], ['annoyance'], ['annoyance', 'neutral'], ['annoyance'], ['optimism'], ['desire', 'optimism'], ['neutral'], ['neutral'], ['disapproval'], ['neutral'], ['admiration'], ['neutral'], ['disapproval'], ['neutral'], ['admiration', 'love'], ['neutral'], ['neutral'], ['neutral'], ['neutral'], ['approval', 'curiosity'], ['approval', 'neutral'], ['approval'], ['neutral'], ['neutral'], ['neutral'], ['confusion', 'realization'], ['neutral'], ['neutral'], ['neutral'], ['approval'], ['neutral'], ['neutral'], ['neutral'], ['disapproval', 'neutral'], ['neutral'], ['curiosity'], ['neutral'], ['neutral'], ['approval'], ['neutral'], ['disapproval'], ['neutral'], ['neutral'], ['approval'], ['neutral'], ['fear'], ['fear'], ['disapproval', 'neutral'], ['surprise', 'neutral'], ['admiration', 'approval'], ['neutral'], ['neutral'], ['approval', 'neutral'], ['admiration', 'love'], ['admiration', 'confusion'], ['neutral'], ['realization', 'neutral'], ['sadness'], ['neutral'], ['neutral'], ['approval'], ['realization'], ['sadness'], ['disgust'], ['anger', 'annoyance'], ['confusion', 'neutral'], ['admiration', 'approval'], ['admiration'], ['surprise'], ['approval', 'excitement'], ['approval'], ['approval', 'neutral'], ['neutral'], ['neutral'], ['neutral'], ['joy'], ['neutral']]

def process(res_list):
    res_dict = {}
    
    for i in res_list:
        if len(i)==0:
            continue
        if i[0] in res_dict:
            res_dict[i[0]]+=1
        else:
            res_dict[i[0]] = 1
        
    res_dict = sorted(res_dict.items(),key=lambda x:x[1],reverse=True)
    res_name_all = [i[0] for i in res_dict]
    res_name_top_5 = ','.join(res_name_all[:5])
    res_name_top_all = ','.join(res_name_all)
    return res_name_top_5, res_name_top_all


tokenizer = BertTokenizer.from_pretrained("monologg/bert-base-cased-goemotions-original")
model = BertForMultiLabelClassification.from_pretrained("monologg/bert-base-cased-goemotions-original")

goemotions = MultiLabelPipeline(
    model=model,
    tokenizer=tokenizer,
    threshold=0.3,
    device=0,
)

texts = [
    "Hey that's a thought! Maybe we need [NAME] to be the celebrity vaccine endorsement!",
    "it’s happened before?! love my hometown of beautiful new ken 😂😂",
    "I love you, brother.",
    "Troll, bro. They know they're saying stupid shit. The motherfucker does nothing but stink up libertarian subs talking shit",
]

excel_dir = './媒体数据'
res_dir = excel_dir+'_res'
os.makedirs(res_dir, exist_ok=True)

name_used = os.listdir(excel_dir)



for name in name_used:
    print(name)
    excel_path = os.path.join(excel_dir, name)
    
    df = pd.read_excel(excel_path)
    df['res'] = [0]*len(df)
    df['res_all'] = [0]*len(df)
    txt_path = os.path.join(res_dir, name[:-5]+'.txt')
    f = open(txt_path, 'a')
    
    # content = df["内容"].to_list()
    content = df["code"].to_list()
    
    for i in tqdm(range(len(df))):
            
        res_all = []
        
        content_now = str(content[i]).strip().replace('&',' ').replace('#',' ')
        
        # print(content_now)
        content_split=content_now.split('.')
        # content_split=content_now.split('.')
        
        
        for idx, j in enumerate(content_split):
            
            labels = goemotions(j)[0]['labels']
            res_all.append(labels)
        # print(res_all)
        res_name_top_5, res_name_top_all = process(res_all)
        print(name,res_name_top_5)
        f.write(res_name_top_5 +'  '+res_name_top_all+'\n')
        
    # df.to_excel(os.path.join(res_dir, name))
    

