# AnimeGANv3   

Paper Title: A Novel Double-Tail Generative Adversarial Network for Fast Photo Animation.
## Let's use AnimeGANv3 to produce our own animation.
<!--
<div style="display:none">
<h3>[project page](https://github.com/TachibanaYoshino/AnimeGANv3/tree/master) | [paper](https://github.com/TachibanaYoshino/AnimeGANv3/tree/master)<h3/>        
</div> -->


**Updates**    
* `2023-11-23` The code and the [manuscript](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/doc/AnimeGANv3_manuscript.pdf) are released. 🦃   
* `2023-10-31` Added three new styles of AnimeGANv3: Portrait  to Cute, 8bit and Sketch-0 style. :ghost:   
* `2023-09-18` Added a new AnimeGANv3 model for Face to Kpop style.     
* `2023-01-16` Added a new AnimeGANv3-photo.exe for the inference of AnimeGANv3's onnx model.     
* `2023-01-13` Added a new AnimeGANv3 model for Face to comic style.     
* `2022-12-25` Added the tiny model (2.4MB) of [~~Nordic myth style~~]() and USA style 2.0. It can go upto 50 FPS on iphone14 with 512*512 input. :santa:       
* `2022-11-24` ~~Added a new AnimeGANv3 model for Face to Nordic myth style.~~ 🦃      
* `2022-11-06` Added a new AnimeGANv3 model for Face to Disney style **V1.0**. :european_castle:     
* `2022-10-31` Added a new AnimeGANv3 model for Face to USA cartoon and Disney style **V1.0**. :jack_o_lantern:    
* `2022-10-07` The USA cartoon Style of AnimeGANv3 is integrated to [**ProfileProfile**](https://apps.apple.com/in/app/profileprofile/id1636884362
) with [Core ML](https://developer.apple.com/documentation/coreml). Install it by the Apple Store and have a try.        
* `2022-09-26` [Official online demo](https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3) is integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3)     
* `2022-09-24` Added a new great AnimeGANv3 model for Face to USA cartoon Style.    
* `2022-09-18` Update a new AnimeGANv3 model for Photo to Hayao Style.    
* `2022-08-01` Added a new AnimeGANv3 onnx model [**(Colab)**](https://www.patreon.com/posts/new-animeganv3-69895469?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator) for Face to [Arcane](https://www.netflix.com/sg/title/81435684) style.    
* `2022-07-13` Added a new AnimeGANv3 onnx model [**(Colab)**](https://colab.research.google.com/drive/1XYNWwM8Xq-U7KaTOqNap6A-Yq1f-V-FB?usp=sharing) for Face to portrait sketch.
* `2021-12-25` The paper of AnimeGANv3 will be released in 2022.:christmas_tree:  
---------  

## Usage
       
* Official online demo is released in [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3).      

* Download this repository and use AnimeGANv3's [UI tool](https://github.com/TachibanaYoshino/AnimeGANv3_gui.exe) and pre-trained *.onnx to turn your photos or videos into anime.:blush:    

* Installation
  1. Clone repo  
      ```bash  
      git clone https://github.com/TachibanaYoshino/AnimeGANv3.git
      cd AnimeGANv3   
      ```
  
  1. Install dependent packages
      ```bash
      pip install -r requirements.txt  
      ```
  1. Inference with *.onnx
      ```bash
      python deploy/test_by_onnx.py -i inputs/imgs/ -o output/results -m deploy/AnimeGANv3_Hayao_36.onnx  
      ```
  1. video to anime with *.onnx
      ```bash
      python tools/video2anime.py -i inputs/vid/1.mp4 -o output/results -m deploy/AnimeGANv3_Hayao_36.onnx  
      ```

## Landscape Demos     
### :fire: Video to anime (Hayao Style)   
<p>
<a href="https://youtu.be/EosubeJmAnE"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 1&color=red"/></a>
<a href="https://youtu.be/5qLUflWb45E"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 2&color=green"/></a>
<a href="https://www.youtube.com/watch?v=iFjiaPlhVm4"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 3&color=pink"/></a>
</p>

  
____
### :art: Photo to Hayao Style    
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/32.jpg)      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/29.jpg)   

<details>
<summary><strong>   more surprise</strong>&emsp;:point_left:</summary>    

![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/33.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/31.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/35.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/4.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/34.jpg)   

</details>    

___
### :art: Photo to Shinkai Style 
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/3.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/4.jpg)  
     
<details>
<summary><strong>   more surprise</strong>&emsp;:point_left:</summary>    

![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/9.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/10.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/8.jpg)  

</details>   
       
___    
## Portrait Style Demos     
**The paper has been completed in 2022. The study of portrait stylization is an extension of the paper.**     
### :art: Face to USA cartoon style     
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_USA/AnimeGANv3_USA_Trump.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_USA/output.jpg)    

___    
### :art: Face to Disney cartoon style     
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Disney/pic.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Disney/output.jpg)  


___    
### :art: Face to USA cartoon + Disney style    
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Trump/AnimeGANv3_Trump_1pic.gif)     
      
<a href="https://youtu.be/vJqQQMRYKh0"><img src="https://img.shields.io/static/v1?label=YouTube&message=AnimeGANv3_Trump style v1.5 &color=gold"/></a>
      
<details>
<summary><strong>   more surprise</strong>&emsp;:point_left:</summary>
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Trump/Trump_output.jpg)      
  
</details> 
      
      
___    
### :art: Face to Arcane style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Arcane/AnimeGANv3_Arcane.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Arcane/AnimeGANv3_Arcane.jpg)   
        
___    
### :art: Face to comic style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_comic/AnimeGANv3_comic.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_comic/AnimeGANv3_comic.jpg)    
     

___    
### :art: Face to Kpop style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Kpop/AnimeGANv3_Kpop.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Kpop/AnimeGANv3_Kpop.jpg)  

___    
### :art: Face to Cute style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Cute/AnimeGANv3_Cute.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Cute/AnimeGANv3_Cute.jpg)    

___    
### :art: Face to 8bit style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_8bit/AnimeGANv3_8bit.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_8bit/AnimeGANv3_8bit.jpg)    

___    
### :art: Face to Sketch-0 style    
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Sketch-0/AnimeGANv3_Sketch-0.jpg)  


___
### :art: Face to portrait sketch   
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYNWwM8Xq-U7KaTOqNap6A-Yq1f-V-FB?usp=sharing)     
      
| input | Face | panoramic image|
| :-: |:-:| :-:|
|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/portrait.jpg" height="60%" width="60%">|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx.png" height="60%" width="60%">|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx1.png" height="60%" width="60%">|
|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/body.jpg" height="60%" width="60%">|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx3.png" height="60%" width="60%" >|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx2.png" height="60%" width="60%">|     
    
<details>
<summary><strong>   more surprise</strong>&emsp;:point_left:</summary>     

![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/face2portrait_sketch.jpg)   

</details>    

___   

## Train

#### 1. Download dataset and pretrained vgg19   
1. [vgg19](https://github.com/TachibanaYoshino/AnimeGAN/releases/download/vgg16%2F19.npy/vgg19_no_fc.npy)   
2. [Hayao dataset](https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/Hayao.tar.gz)   
3. [Shinkai dataset](https://github.com/TachibanaYoshino/AnimeGANv2/releases/download/1.0/Shinkai.tar.gz)   
4. [photo dataset](https://github.com/TachibanaYoshino/AnimeGAN/releases/download/dataset-1/dataset.zip)   

#### 2. Do edge_smooth  
```bash
    cd tools && python edge_smooth.py --dataset Hayao --img_size 256
  ```

#### 3. Do superPixel
```bash
    cd tools && python visual_superPixel_seg_image.py
  ```  

#### 4. Train  
  >  `python train.py --style_dataset Hayao --init_G_epoch 5 --epoch 100`


## Citation   
Consider citing as below if you find this repository helpful to your project:   
```
@article{Liu2024dtgan,
  title={A Novel Double-Tail Generative Adversarial Network for Fast Photo Animation},
  author={Gang Liu and Xin Chen and Zhixiang Gao},
  journal={IEICE Transactions on Information and Systems},
  year={2024}
}
```

## :scroll: License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv3 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.    

## :e-mail: Author  
Asher Chan `asher_chan@foxmail.com`
    
