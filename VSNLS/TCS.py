import sys
import pickle 
import openai
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
api_key = str(sys.argv[1])
openai.api_key = api_key

# Video Caption
ag_caption = pd.read_csv("datasets/AG/Charades_vu17_train.csv")
caption_dict = {}
for k, v in ag_caption.iterrows():
    video_id = v['id'] + ".mp4"
    caption_dict[video_id] = v['descriptions']

# AG Train frame list
with open('datasets/AG/ag_train_id.pkl', 'rb') as f:
    video_frame_dict = pickle.load(f)
    
        
print(f"# Video: {len(video_frame_dict)}")

video_index_list = list(video_frame_dict.keys())
split_action_dict = defaultdict(list)

for p, k in enumerate(tqdm(video_index_list)):
    v = caption_dict[k]
    split_caption = v.split(';')
    
    for input_caption in split_caption:
        prompt=f'''
        In this task, you are given a video caption describing a video. Considering the words that indicate the order of events (e.g., then, while, before, and after), your job is to split multiple compositional sentences from the given video caption and list them in chronological order. Note that you should specify the objects for the pronouns used in each of these sentences. 
        Input: The person is turning on the stove. They then begin to stir some food and after that they pick up a camera and look at it. 
        Output: The person is turning on the stove. >> The person stirs some food. >> The person picks up a camera. >> The person looks at a camera.
        Input: A person is sitting in bed texting on a phone while holding a blanket. The person puts the phone down and pulls the blanket up. 
        Output: A person is sitting in a bed and texting on a phone while holding a blanket. >> The person puts the phone down. >> The person pulls the blanket up.
        Input: A person picks up a phone and enters the bathroom through a doorway while talking on the phone. The person puts on shoes and picks up clothes while laughing and dresses before walking out of the room. 
        Output: A person picks up a phone. >> A person enters the bathroom through a doorway while talking on the phone. >> The person puts on shoes >> The person picks up clothes while laughing >> The person dresses clothes >> The person walks out of the room.
        Input: A person is sitting on a toilet, picks up a phone and battery that are on the ground, puts the battery into the phone, takes off a jacket, then stands and takes selfies against the bathroom door. 
        Output: A person is sitting on a toilet. >> A person picks up a phone and battery that are on the ground. >> A person takes off a jacket. >> A person stands and takes selfies against the bathroom door.
        Input: A person is undressing, picks up a towel and cleans some glasses before taking a drink. 
        Output: A person is undressing. >> A person picks up a towel. >> A person cleans some glasses. >> A person takes a drink some glasses.
        Input: Person pulls out phone and begins playing with it then sets it down and pulls the blanket further up. 
        Output: Person pulls out phone. >> Person plays with the phone. >> Person sets the phone down. >> Person pulls the blanket further up.
        Input: A person watching television and eating a sandwich while laying on the floor and reading book,after a while the person gets up to grab a box.
        Output: A person watches television and eats a sandwich while laying on the floor. >> A person reads a book. >> A person gets up to grab a box.
        Input: A person walks to a pantry, takes out some clothes from it, tosses one on the floor, and puts on another after taking it off again. 
        Output: A person walks to a pantry. >> A person takes out some clothes from a pantry. >> A person tosses a cloth on the floor. >> A person puts on a cloth. >> A person takes a cloth off.
        Input: {input_caption}. 
        '''
        stop = False
        while not stop:
            try:
                completion = openai.ChatCompletion.create(
                model='gpt-3.5-turbo', 
                messages=[{"role": "user", "content": prompt}],
                temperature=0
                )
                response = completion.choices[0].message.content
                stop=True
            except:
                pass
            
        split_action_dict[k].append(response)

split_action_preprocessed_dict = defaultdict(list)
for k, s_a in split_action_dict.items():
    for v in s_a:
        temp_split_caption_list = []
        action_sequences = v.split("Output")[1][3:].strip()
        action_sequences = action_sequences.split(">>")
        for a_s in action_sequences:
            temp_split_caption_list.append(a_s.strip().strip("'").strip('"').strip("."))
        split_action_preprocessed_dict[k].append(temp_split_caption_list)

# Revise Error Case
split_action_preprocessed_dict['1ECM2.mp4'][1][0] = 'A person is holding a broom while walking in a closet'
split_action_preprocessed_dict['ESAIY.mp4'][1][1] = 'A person is taking medicine'
split_action_preprocessed_dict['SOTQ1.mp4'][1][1] = 'The other person is sitting cross-legged on a table apparently doing homework'
split_action_preprocessed_dict['X1624.mp4'][1][2] = 'A person takes off a jacket while holding a broom'
       
with open(f"datasets/AG/split_action_dict.pkl", 'wb') as f:
    pickle.dump(split_action_preprocessed_dict, f)