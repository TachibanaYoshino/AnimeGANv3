# AnimeGANv3   

Paper Title: A Novel Double-Tail Generative Adversarial Network for Fast Photo Animation.
## Let's use AnimeGANv3 to produce our own animation.

<div align="center">    
           
[![manuscript](https://img.shields.io/badge/manuscript-PDF-gold?logo=googledocs&logoColor=gold)](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/doc/AnimeGANv3_manuscript.pdf)
[![Paper](https://img.shields.io/badge/cs.CV-Paper-violet?logo=docusign&logoColor=violet)](https://www.jstage.jst.go.jp/article/transinf/E107.D/1/E107.D_2023EDP7061/_pdf/-char/en)
[![Project Page](https://img.shields.io/badge/Project-Website-green?logo=googlechrome&logoColor=green)](https://tachibanayoshino.github.io/AnimeGANv3/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Leaderboard-40D1F5)](https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3)
[![Video](https://img.shields.io/badge/YouTube-Video-b31b1b?logo=youtube&logoColor=red)](https://youtu.be/EosubeJmAnE)
[![twitter](https://img.shields.io/badge/twitter-Asher-1D9BF0?logo=x&logoColor=#1D9BF0)](https://twitter.com/asher_9527)
[![LICENSE](https://img.shields.io/badge/license-AnimeGANv3-AB82FF?logo=leagueoflegends&logoColor=AB82FF)](https://github.com/TachibanaYoshino/AnimeGANv3?tab=readme-ov-file#scroll-license)
[![Github](https://img.shields.io/github/stars/TachibanaYoshino/AnimeGANv3?logo=githubsponsors&logoColor=#EA4AAA)](https://github.com/TachibanaYoshino/AnimeGANv3)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://drive.google.com/file/d/1Er23bL36pkr67Q9f1P28BuMP6yZKf-yz/view?usp=sharing)
[![Visitor](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FAnimeGANv3&count_bg=%23FFA500&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=visitors&edge_flat=false)](https://hits.seeyoufarm.com)

</div>       

## 📢 Updates    
* `2024-10-31` Added a new styles of AnimeGANv3: Portrait to Pixar. :jack_o_lantern:    
* `2024-08-28` A [repo](https://github.com/TachibanaYoshino/AnimeGANv3_Portrait_Inference) more suitable for portrait style inference based on the AnimeGANv3 models has been released. Highly recommended.         
* `2023-12-10` Added a new AnimeGANv3 model for Portrait to Oil-painting style. Its onnx is available [here](https://www.patreon.com/posts/animeganv3-s-oil-94445425?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link).     
* `2023-11-23` The code and the [manuscript](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/doc/AnimeGANv3_manuscript.pdf) are released.  🦃   
* `2023-10-31` Added three new styles of AnimeGANv3: Portrait to Cute, 8bit and Sketch-0 style. :ghost:   
* `2023-09-18` Added a new AnimeGANv3 model for Face to Kpop style.     
* `2023-01-16` Added a new AnimeGANv3-photo.exe for the inference of AnimeGANv3's onnx model.     
* `2023-01-13` Added a new AnimeGANv3 model for Face to comic style.     
* `2022-12-25` Added the tiny model (2.4MB) of [~~Nordic myth style~~]() and USA style 2.0. It can go upto 50 FPS on iphone14 with 512*512 input. :santa:       
* `2022-11-24` ~~Added a new AnimeGANv3 model for Face to Nordic myth style.~~  🦃      
* `2022-11-06` Added a new AnimeGANv3 model for Face to Disney style **V1.0**.         
* `2022-10-31` Added a new AnimeGANv3 model for Face to USA cartoon and Disney style **V1.0**.  :jack_o_lantern:    
* `2022-10-07` The USA cartoon Style of AnimeGANv3 is integrated to [**ProfileProfile**](https://apps.apple.com/in/app/profileprofile/id1636884362
) with [Core ML](https://developer.apple.com/documentation/coreml). Install it by the Apple Store and have a try.        
* `2022-09-26` [Official online demo](https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3) is integrated to [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3)     
* `2022-09-24` Added a new great AnimeGANv3 model for Face to USA cartoon Style.    
* `2022-09-18` Update a new AnimeGANv3 model for Photo to Hayao Style.    
* `2022-08-01` Added a new AnimeGANv3 onnx model [**(Colab)**](https://www.patreon.com/posts/new-animeganv3-69895469?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator) for Face to [Arcane](https://www.netflix.com/sg/title/81435684) style.    
* `2022-07-13` Added a new AnimeGANv3 onnx model [**(Colab)**](https://colab.research.google.com/drive/1XYNWwM8Xq-U7KaTOqNap6A-Yq1f-V-FB?usp=sharing) for Face to portrait sketch.
* `2021-12-25` The paper of AnimeGANv3 will be released in 2022.  :christmas_tree:  
---------  

## 🎮 Usage
       
* Official online demo is released in [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/TachibanaYoshino/AnimeGANv3).      

* Download this repository and use AnimeGANv3's [UI tool](https://github.com/TachibanaYoshino/AnimeGANv3_gui.exe) and pre-trained *.onnx to turn your photos into anime. :blush:    

* 🛠️ Installation
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
<br/>    

## 🚀 Landscape Demos     
### :fire: Video to anime (Hayao Style)   
<p>
<a href="https://youtu.be/EosubeJmAnE"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 1&color=red"/></a>
<a href="https://youtu.be/5qLUflWb45E"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 2&color=green"/></a>
<a href="https://www.youtube.com/watch?v=iFjiaPlhVm4"><img src="https://img.shields.io/static/v1?label=YouTube&message=video 3&color=pink"/></a>
</p>   

____     

### :art: Photo to Hayao Style    
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/4.jpg)      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/29.jpg)   

<details>
<summary><strong>   more surprise</strong>&emsp;👈</summary>    

![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/33.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/31.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/35.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/32.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Hayao/34.jpg)   
</details>    

___   
<br/>   

### :art: Photo to Shinkai Style 
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/3.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/4.jpg)  
     
<details>
<summary><strong>   more surprise</strong>&emsp;👈 </summary>    

![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/9.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/10.jpg)   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/11.jpg)  
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Shinkai/8.jpg)  
</details>   

___
<br/>   

## 🚀 Portrait Style Demos     
**The paper has been completed in 2022. The study of portrait stylization is an extension of the paper.**     

<details>
<summary><strong>   Some exhibits </strong>&emsp;👈</summary>   
       
### :art: Face to USA cartoon style     
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_USA/AnimeGANv3_USA_Trump.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_USA/output.jpg)    

https://private-user-images.githubusercontent.com/36946777/377949823-9644b1f5-78a4-4dcd-9da0-0186fbf5ab94.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjQ1ODUsIm5iZiI6MTczMDM2NDI4NSwicGF0aCI6Ii8zNjk0Njc3Ny8zNzc5NDk4MjMtOTY0NGIxZjUtNzhhNC00ZGNkLTlkYTAtMDE4NmZiZjVhYjk0Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NDQ0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTE1YWRhYTU4ZWUzMmE1ZDk5YWZjNjliNDA1ZDI0YmJjMzQ0MGYwMTNkYmVmMWZkYWYxMGQzZGQyNmFhNDU3MGEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.YeMCbc9leC45euFBXeOUVmrTk1Zp-voro5_tkoc1u28

___    
### :art: Face to Disney cartoon style     
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Disney/pic.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Disney/output.jpg)  

| v1.9 | v2.0 |
|:-:|:-:| 
|<video  src="https://private-user-images.githubusercontent.com/36946777/377891000-9cc111e9-8a1d-4c22-b0d0-0c430aca98d5.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjU2NTksIm5iZiI6MTczMDM2NTM1OSwicGF0aCI6Ii8zNjk0Njc3Ny8zNzc4OTEwMDAtOWNjMTExZTktOGExZC00YzIyLWIwZDAtMGM0MzBhY2E5OGQ1Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA5MDIzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWFkMGY2MTA4MjVmZjRlOGE4NDAxZmQzYmJmOTZiY2QzZjNiOTdhM2QyMzUzZjI0N2NiNTAyNmMxMzZiNzJkYjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.MWvyDwwPyCtQ8_d2FqPPvjwmy7PLw4CoS4zPGkgm3eQ" type="video/mp4"> </video>|<video  src="https://private-user-images.githubusercontent.com/36946777/377891125-8d7f2be8-14d7-4447-b376-6b0f44a8b8fd.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjU2NTksIm5iZiI6MTczMDM2NTM1OSwicGF0aCI6Ii8zNjk0Njc3Ny8zNzc4OTExMjUtOGQ3ZjJiZTgtMTRkNy00NDQ3LWIzNzYtNmIwZjQ0YThiOGZkLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA5MDIzOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTdlMDY5MDZlNjgzNjY1YzE1NzhmMjFhNWI3NDEyYzAyZGFiYjViYmFkNTI0NGJiMzNiMTA0ZWU5ZmYxMmVlZjEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.qopnTZwTFVGqsOPmhStZNCqiNEsK4fBSlJQ5ScG1nmk" type="video/mp4"> </video>| 
___    
### :art: Face to USA cartoon + Disney style    
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Trump/AnimeGANv3_Trump_1pic.gif)     
      
<a href="https://youtu.be/vJqQQMRYKh0"><img src="https://img.shields.io/static/v1?label=YouTube&message=AnimeGANv3_Trump style v1.5 &color=gold"/></a>
      
<details>
<summary><strong>   more surprise</strong>&emsp;👈</summary>
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Trump/Trump_output.jpg)      
</details>   

___    

### :art: Face to Arcane style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Arcane/AnimeGANv3_Arcane.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Arcane/AnimeGANv3_Arcane.jpg)   

https://private-user-images.githubusercontent.com/36946777/377960626-ab082d32-cc77-4c89-92c1-6a50cfa6a77b.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjQ1ODUsIm5iZiI6MTczMDM2NDI4NSwicGF0aCI6Ii8zNjk0Njc3Ny8zNzc5NjA2MjYtYWIwODJkMzItY2M3Ny00Yzg5LTkyYzEtNmE1MGNmYTZhNzdiLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NDQ0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWQ1ZTIwZjRiNTkxNGFiYzA1OGYyYmU4YTkwNTk2NmRiNTI2ZjgyMDk5ZjdkYWE1NTljMmE5Yzk2NDRiNDc3ODUmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.KwzpFZ3-OWOivQ2r8F8muhpO7lyllnfSRmJ4N9UlUAg

___    
### :art: Portrait to comic style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_comic/AnimeGANv3_comic.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_comic/AnimeGANv3_comic.jpg)    

https://private-user-images.githubusercontent.com/36946777/378193688-3e999a8e-a331-46f6-863c-c01fd50591c8.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjQ1ODUsIm5iZiI6MTczMDM2NDI4NSwicGF0aCI6Ii8zNjk0Njc3Ny8zNzgxOTM2ODgtM2U5OTlhOGUtYTMzMS00NmY2LTg2M2MtYzAxZmQ1MDU5MWM4Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NDQ0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTVhNTFiNzVjODk1YjA4YzFjODRmYWQ4NGExOWJkNzdkYjc0ZGMxZWNhOGZjOGNhMDAzZjJkNGY0NjUzODk2NGQmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.0460wu-mwqsRRLXux_RmGL_Er_faR5r1zxxvWD3k0HY   

___    
### :art: Face to Kpop style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Kpop/AnimeGANv3_Kpop.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Kpop/AnimeGANv3_Kpop.jpg)  

https://private-user-images.githubusercontent.com/36946777/381846668-3a59537c-fff2-4c86-8462-d53b07ff596b.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjQ1ODUsIm5iZiI6MTczMDM2NDI4NSwicGF0aCI6Ii8zNjk0Njc3Ny8zODE4NDY2NjgtM2E1OTUzN2MtZmZmMi00Yzg2LTg0NjItZDUzYjA3ZmY1OTZiLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NDQ0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTViZTMxNDU5MDRiYTI4OTMyMTZiMzVlNmNlN2MzZDE1ZDQwMTc1ZjczNWQxMWE3NjE5NDliYWVjZmQ0YWM4NDMmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.HMk5vGhi6jVo2z0qmXytGSes_YDuh9Iu6pbrKf51lkM

___    
### :art: Portrait to Oil-painting style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_oil-painting/AnimeGANv3_oil-painting.gif)     

<details>
<summary><strong>   more surprise</strong>&emsp;👈 </summary>

![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_oil-painting/AnimeGANv3_oil-painting.jpg)     
</details>  

___    
### :art: Portrait to Cute style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Cute/AnimeGANv3_Cute.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Cute/AnimeGANv3_Cute.jpg)    

https://private-user-images.githubusercontent.com/36946777/377891282-0b105ee7-8116-4456-931c-ec196200e288.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjQ1ODUsIm5iZiI6MTczMDM2NDI4NSwicGF0aCI6Ii8zNjk0Njc3Ny8zNzc4OTEyODItMGIxMDVlZTctODExNi00NDU2LTkzMWMtZWMxOTYyMDBlMjg4Lm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NDQ0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWZiYzE0ODBmMWI3OGI4MGU1MGI2Yjg5MDFkY2I1ZjM4ZTAwNWE2MTU1YTY3NzA1MzM1YWU5ODMzZGM0ZmQ1ZWEmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.vsEk5C8_pm2oyHN4QwSIefz7zvbGx8qMMK6k8dT0TTY   
___  
### :art: Portrait to Pixar style    

![](https://private-user-images.githubusercontent.com/36946777/381880672-fc113b82-2a07-434a-9e17-1d3009bd0c5a.jpg?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjUzMTEsIm5iZiI6MTczMDM2NTAxMSwicGF0aCI6Ii8zNjk0Njc3Ny8zODE4ODA2NzItZmMxMTNiODItMmEwNy00MzRhLTllMTctMWQzMDA5YmQwYzVhLmpwZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NTY1MVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPThlY2I3NjgxNzIzY2EwNzViNGFiZWFlZmYyMmFlMjA5MWM1YmE5MTQ4OWQ0YjVlMjQ5MDgzYWNkY2VjYTg2MzImWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.devmqkSBu-HdVPBPHBBtN_1SIJrcwO_kVuZmYDHrmKo) 

      
https://private-user-images.githubusercontent.com/36946777/381869713-d9c4e931-3b3c-4b03-9531-63d9e391b4df.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjQ1ODUsIm5iZiI6MTczMDM2NDI4NSwicGF0aCI6Ii8zNjk0Njc3Ny8zODE4Njk3MTMtZDljNGU5MzEtM2IzYy00YjAzLTk1MzEtNjNkOWUzOTFiNGRmLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NDQ0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTAyOWJkZmRmYzk0N2I4ZTA1YmJkNzRmMDE5ZjIwNmZhNzU0OWJkMmNlYmZkZTc0NmEzMWNlZDdmMTBkMjQ4ZTYmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.1gOxYkwHz3v-dVt852g_3y2ROqUb0ItuH7O911-UBcQ   

___    
  
### :art: Portrait to Sketch-0 style    
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Sketch-0/AnimeGANv3_Sketch-0.jpg)  

https://private-user-images.githubusercontent.com/36946777/381869617-ed3f3511-4583-41d8-aad9-e47fdd2f5c32.mp4?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MzAzNjQ1ODUsIm5iZiI6MTczMDM2NDI4NSwicGF0aCI6Ii8zNjk0Njc3Ny8zODE4Njk2MTctZWQzZjM1MTEtNDU4My00MWQ4LWFhZDktZTQ3ZmRkMmY1YzMyLm1wND9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMzElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDMxVDA4NDQ0NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPWMwMzJjNDkxNTg3OTY3ZjAyM2RmNDZhMWM0ZDY0NzQ2Yjk3NmVjZmQyMDNmOGY1MjVkMTU5N2E4ZjNmODZkY2MmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.iVgU45BsuYGLoVECbqT3DE_9yzlFqKnFedu4mJPUA7g

___  

### :art: Portrait to 8bit style   
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_8bit/AnimeGANv3_8bit.gif)     
      
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_8bit/AnimeGANv3_8bit.jpg)    

___
### :art: Face to portrait sketch   
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XYNWwM8Xq-U7KaTOqNap6A-Yq1f-V-FB?usp=sharing)     
      
| input | Face | panoramic image|
| :-: |:-:| :-:|
|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/portrait.jpg" height="60%" width="60%">|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx.png" height="60%" width="60%">|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx1.png" height="60%" width="60%">|
|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/body.jpg" height="60%" width="60%">|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx3.png" height="60%" width="60%" >|<img src="https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/output_onnx2.png" height="60%" width="60%">|     
    
<details>
<summary><strong>   more surprise</strong>&emsp;👈</summary>     
       
![](https://github.com/TachibanaYoshino/AnimeGANv3/blob/master/results/AnimeGANv3_Face2portrait_sketch/face2portrait_sketch.jpg)    
       
</details>    

</details>  

<br/>   

## 🔨 Train

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
```bash
    python train.py --style_dataset Hayao --init_G_epoch 5 --epoch 100
  ```  

<br/>   

## ✒️ Citation   
Consider citing as below if you find this repository helpful to your project:   
```bibtex
@article{Liu2024dtgan,
  title={A Novel Double-Tail Generative Adversarial Network for Fast Photo Animation},
  author={Gang LIU and Xin CHEN and Zhixiang GAO},
  journal={IEICE Transactions on Information and Systems},
  volume={E107.D},
  number={1},
  pages={72-82},
  year={2024},
  doi={10.1587/transinf.2023EDP7061}
}
```

## :scroll: License  
This repo is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications. Permission is granted to use the AnimeGANv3 given that you agree to my license terms. Regarding the request for commercial use, please contact us via email to help you obtain the authorization letter.    

## :e-mail: Author  
Asher Chan `asher_chan@foxmail.com`
    
