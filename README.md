# Car-Lane-Detection
This repo was designed to use in [Baidu Car Lane Detection Challenge](https://aistudio.baidu.com/aistudio/competition/detail/5).

## Preparation
1. Download the dataset from [here](https://aistudio.baidu.com/aistudio/competition/detail/5)
2. Setup the data folder in this structure
```python
   Image/
     road02/
       Record002/
         Camera 5/
           ...
         Camera 6
       Record003
       ....
     road03
     road04
   Label/
     Label_road02/
      Label
       Record002/
         Camera 5/
          ...
         Camera 6
       Record003
       ....
     Label_road03
     Label_road04
```
3. Change the path in make_list.py and run
```python
image_dir = Path('yourPath/Image_Data/')
label_dir = Path('yourPath/Label/')
save_path = Path('yourPath/data_list/')
```
```
python -make_list.py
```
After this step, train dataset, validate dataset will be created.

## Train
1. Change the GPU setting in train.py
```python
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device_list = [0]
```
2. Run train.py
```
python -train.py
```

## Predict
After training, model will be stored under logs. Run predict.py
```
python -predict.py
```
