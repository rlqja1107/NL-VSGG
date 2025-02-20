import sys
import re
import pickle
import openai
from tqdm import tqdm
from collections import defaultdict
api_key = str(sys.argv[1])
openai.api_key = api_key

with open("datasets/AG/split_action_dict.pkl", 'rb') as f:
    split_action_preprocessed_dict = pickle.load(f)

# AG Train frame list
with open('datasets/AG/ag_train_id.pkl', 'rb') as f:
    video_frame_dict = pickle.load(f)


obj_class = ['__background__']
with open("datasets/AG/object_classes.txt", 'r') as f:
    for i in f.readlines():
        obj_class.append(i.strip('\n'))
obj_class[9] = 'cabinet'
obj_class[11] = 'cup'
obj_class[23] = 'paper'
obj_class[24] = 'phone'
obj_class[31] = 'sofa'


action_class = ['looking at', 'not looking at', 'unsure', 'above', 'beneath', 'in front of', 'behind', 'on the side of', 'in', 'carrying', 'covered by', 'drinking from', 'eating', 'have it on the back', 'holding', 'leaning on', 'lying on', 'not contacting', 'other relationship', 'sitting on', 'standing on', 'touching', 'twisting', 'wearing', 'wiping', 'writing on']    


video_index_list = list(split_action_preprocessed_dict.keys())

extracted_triplet_dict = defaultdict(list)
for video_index in tqdm(video_index_list):
    split_action = split_action_preprocessed_dict[video_index]
    for s_a in split_action:
        sentence_list = []
        for sentence in s_a:
            sentence_list.append(sentence.strip().strip('"').strip("'").strip('.'))
        prompt = f'''
        In this task, you are given an input sentence. Based on the given sentence, your job is to extract meaningful triplets formed as <subject, predicate, object>, where the object is a lexeme in the predefined entity lexicon, and the predicate is a lexeme in the predefined predicate lexicon. Please note that the subject of the given sentence is a person. Therefore, if the subject is omitted, consider it as a person. 

        The predefined entity lexicon containing 36 lexemes is numbered as follows: 1.person 2.bag 3.bed 4.blanket 5.book 6.box 7.broom 8.chair 9.cabinet 10.clothes 11.cup 12.dish 13.door 14.doorknob 15.doorway 16.floor 17.food 18.groceries 19.laptop 20.light 21.medicine 22.mirror 23.paper 24.phone 25.picture 26.pillow 27.refrigerator 28.sandwich 29.shelf 30.shoe 31.sofa 32.table 33.television 34.towel 35.vacuum 36.window.

        The predefined predicate lexicon containing 26 lexemes is numbered as follows: 1.looking at 2.not looking at 3.unsure 4.above 5.beneath 6.in front of 7.behind 8.on the side of 9.in 10.carrying 11.covered by 12.drinking from 13.eating 14.have it on the back 15.holding 16.leaning on 17.lying on 18.not contacting 19.other relationship 20.sitting on 21.standing on 22.touching 23.twisting 24.wearing 25.wiping 26.writing on. 

        However, if there is no semantically similar lexeme in the predefined lexicon, please replace it with 0.None for entity and predicate.
        Let's take a few examples to understand how to extract meaningful triplets. 

        Input: The person kneels in front of the sink and wipes down the cabinet. 
        Output: Step 1. Triplets extracted from the sentence are <person, kneels in front of, sink> and <person, wipes down, cabinet>. Step 2. Triplets aligned with the predefined entity/predicate lexicons are <1.person, 6.in front of, 0.None> and <1.person, 25.wiping, 9.cabinet>.
        Input: The person puts the glass on a shelf in front of a window. 
        Output: Step 1: Triplets extracted from the sentence are <person, puts, glass> and <person, in front of, window>. Step 2: Triplets aligned with the predefined entity/predicate lexicons are <1.person, 15.holding, 11.cup> and <1.person, 6.in front of, 36.window>.
        Input: Another person is standing in front of them with a broom. 
        Output: Step 1: A triplet extracted from the sentence is <person, standing in front of, broom>. Step 2: Triplet aligned with the predefined entity/predicate lexicons are <1.person, 21.standing on, 7.broom> and <1.person, 6.in front of, 7.broom>.
        Input: The person sees another person eating a snack. 
        Output: Step 1: Triplets extracted from the sentence are <person, sees, person> and <person, eating, snack>. Step 2: Triplets aligned with the predefined entity/predicate lexicons are <1.person, 1.looking at, 1.person> and <1.person, 13.eating, 17.food>.
        Input: They write something on some paper. 
        Output: Step 1: A triplet extracted from the sentence is <They, write, paper>. Step 2: A triplet aligned with the predefined entity/predicate lexicons is <1.person, 26.writing on, 23.paper>.
        Input: A person opens the refrigerator and looks inside of it. 
        Output: Step 1: Triplets extracted from the sentence are <person, opens, refrigerator> and <person, looks inside, refrigerator>. Step 2: Triplets aligned with the predefined entity/predicate lexicons are <1.person, 0.None, 27.refrigerator> and <1.person, 1.looking at, 27.refrigerator>.
        Input: A person comes in and takes off jacket and puts it on the back of the chair. 
        Output: Step 1: Triplets extracted from the sentence are <person, takes off, jacket>, <person, puts, jackets>, and <person, on the back of, chair>. Step 2: Triplets aligned with the predefined entity/predicate lexicons are <1.person, 0.None, 10.clothes>, <1.person, 15.holding, 10.clothes>, and <1.person, 14.have it on the back of, 8.chair>.
        Please output the answer of following {len(sentence_list)} input sentences. 
        '''
        for c in sentence_list:
            prompt += f"Input: {c}. Output: "
            
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
                extracted_triplet_dict[video_index].append(response)
            except:
                pass

# Pre-process
extracted_triplet_preprocessed_dict = {}
triplet_cnt = 0
error_id_list = []
for k, response_list in extracted_triplet_dict.items():
    extracted_triplet_preprocessed_dict[k] = {}
    extracted_triplet_preprocessed_dict[k]['frame_list'] = video_frame_dict[k]
    extracted_triplet_preprocessed_dict[k]['split_sentence'] = split_action_preprocessed_dict[k]
    extracted_triplet_preprocessed_dict[k]['triplets'] = [[[] for _ in range(len(split_action_preprocessed_dict[k][i]))] for i in range(len(split_action_preprocessed_dict[k]))]
    
    input_sentence_list = []
    output_triplets_list = []
    break_flag = True
    for r_l in response_list:
        temp_input_sentence_list = []
        temp_output_triplet_list = []
        split_sentence = r_l.split('Input')
        for s_s in split_sentence[1:]:
            s_s = s_s.split("Step")
            input_sentence = s_s[0][1:].split("Output")[0].strip('\n').strip().strip("'").strip('"').strip('.')
            
            idx = -1; idx2 = -1
            for ith, s_a in enumerate(split_action_preprocessed_dict[k]):
                for jth, s_c in enumerate(s_a):
                    if input_sentence.strip('.').lower() == s_c.lower():
                        idx = ith
                        idx2 = jth
            if idx == -1:
                print(k)
                
            temp_input_sentence_list.append(input_sentence)
            triplet_per_caption = []
            try:
                output_triplet = s_s[2].split('<')
            except:
                break_flag = False
                pass
            output_triplet_per_caption = []
            for o in output_triplet[1:]:
                o = o.split(',')
                if len(o) < 3: continue
                sub = o[0].strip(); action = o[1].strip(); obj = o[2].strip().strip("\n\n").split('>')[0].strip()
                sub = re.sub(r'\b\d+.\s*', '', sub); action = re.sub(r'\b\d+.\s*', '', action); obj = re.sub(r'\b\d+.\s*', '', obj)
                if action == 'None': action = 'unsure'
                if sub in obj_class and obj in obj_class and action in action_class:
                    extracted_triplet_preprocessed_dict[k]['triplets'][idx][idx2].append((sub, action, obj))
                    output_triplet_per_caption.append((sub, action, obj))
                    triplet_cnt += 1
            temp_output_triplet_list.append(output_triplet_per_caption)
        output_triplets_list.append(temp_output_triplet_list)
        input_sentence_list.append(temp_input_sentence_list)
    if not break_flag:
        del extracted_triplet_preprocessed_dict[k]
        continue
    
    delete = True
    no_triplet = True
    for i_s in input_sentence_list:
        for i in i_s:
            if len(i) > 0:
                delete=False
    for o_t in output_triplets_list:
        for o in o_t:
            if len(o) > 0:
                delete=False
                no_triplet = False
    if delete or no_triplet:
        error_id_list.append(k)
        del extracted_triplet_preprocessed_dict[k]
        continue
    if len(input_sentence_list) != len(output_triplets_list):
        del extracted_triplet_preprocessed_dict[k] 

print(f"# Triplet: {triplet_cnt}")

with open(f"datasets/AG/triplets_LLM4SGG.pkl", 'wb') as f:
    pickle.dump(extracted_triplet_preprocessed_dict, f)
    