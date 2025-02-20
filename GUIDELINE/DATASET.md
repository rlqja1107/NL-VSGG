## Download Dataset  

In the following [link](https://prior.allenai.org/projects/charades), please download `Data(scaled to 480p, 13GB)` for raw videos and `Annotions & Evaluation Code (3 MB)` for video captions. Then, please put them in *datasets/AG*. We use **Charades_v1_train.csv** file containing video captions. 

For frames, we follow previous work [Action Genome](https://github.com/JingweiJ/ActionGenome?tab=readme-ov-file). Specifically, we use **frame_list.txt** in [link](https://drive.google.com/drive/folders/1LGGPK_QgGbh9gH9SDFv_9LIhBliZbZys) for frame list. 
Regarding the action genome's meta data, please download `object_bbox_and_relationship_filtersmall.pkl`, `person_bbox.pkl`, `object_classes.txt`, and `relationship_classes.txt` in [Action Genome](https://github.com/JingweiJ/ActionGenome?tab=readme-ov-file).  

### Dump frames   

We use the same code with [Action Genome](https://github.com/JingweiJ/ActionGenome) to dump frames. 

* Video → Multiple Frames

``` python  
python VSNLS/data_preprocess/dump_frames.py
```

### Data Directory  

This provides an overview of the dataset directory structure.  

```
root  
├── datasets 
│   ├── AG     
│   │   │── frames    
|   │   │    │── {video_id}
|   │   │    │       └── *.png  
│   │   │── video 
|   │   │    └── *.mp4        
│   │   ├── Charades_v1_train.csv (Video caption)
│   │   ├── frame_list.txt (frame list)
│   │   ├── ag_train_id.pkl
│   │   ├── ag_test_id.pkl
│   │   ├── object_bbox_and_relationship_filtersmall.pkl
│   │   ├── person_bbox.pkl
│   │   ├── object_classes.txt
│   │   ├── relationship_classes.txt
│   │   ├── ag_img_info_train.pkl
│   │   └── ag_img_info_test.pkl
│   │
│   ├── ag_to_oi_word_map_synset.npy (Provided by PLA)
│   ├── oi_to_ag_word_map_synset.npy (Provided by PLA)
│   └── VG-SGG-dicts-vgoi6-clipped.json (Provided by PLA)
```  
* PLA: [link](https://github.com/zjucsq/PLA)  
* [ag_train_id.pkl](https://drive.google.com/file/d/1h_kcdJdfwnPyHUWugS8nxHPeAno_7aPv/view?usp=sharing), [ag_test_id.pkl](https://drive.google.com/file/d/1g2xru3KayaIyCuJhxKsYpIe4hgNr3yI9/view?usp=sharing): Following [STTran](https://github.com/yrcong/STTran), we collect video index list with frames.

* [ag_img_info_train.pkl](https://drive.google.com/file/d/1eZw9ElQBbAFAnBkHF9-pjnsfW55e8SMd/view?usp=sharing), [ag_img_info_test.pkl](https://drive.google.com/file/d/14DvE5pWBWPlBG6yUnorIWt3rMztntetq/view?usp=sharing): Transformed images' width and height  
Please refer to [VSNLS/data_preprocess/extract_ag_img_info.py](../VSNLS/data_preprocess/extract_ag_img_info.py)
