## Package & Repository Requirement  

Please install required packages or repositories step by step.  

``` python  
conda create -n vsgg python=3.8.0
conda activate vsgg

# pytorch version: 1.10.1
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/cu111/torch_stable.html

pip install opencv-python tqdm pandas scikit-learn ftfy regex ninja Cython tensorboardX termcolor easydict pyyaml matplotlib yacs timm einops pycocotools cityscapesscripts h5py dill
pip install openai==0.27.8
pip install numpy==1.21.6

# Clone VL model 
git clone https://github.com/SivanDoveh/DAC.git 

# Clone SGG repo
git clone https://github.com/microsoft/scene_graph_benchmark

# Setup
cd lib/draw_rectangles
python setup.py build_ext --inplace
cd ..
cd fpn/box_intersections_cpu
python setup.py build_ext --inplace
cd ../../..

cd scene_graph_benchmark
python setup.py build develop
cd ..

cd fasterRCNN/lib
python setup.py build develop
```   

* Please change `torch._six.PY3` to `torch._six.PY37` in *scene_graph_benchmark/maskrcnn_benchmark/utils/import.py* file.

### Base model

* DAC: [LLM_cp.pt](https://drive.google.com/drive/folders/1DmHeV8oWiMwtkaTH-nruMyjBiuJvcwnv)  
Please put this pre-trained model in *DAC* directory.  