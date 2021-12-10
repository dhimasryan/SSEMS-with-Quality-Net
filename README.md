# Specialized-Speech-Enhancement-Model-Selection-Based-on-Quality-Net
### Introduction ###

Previous studies have shown that a specialized speech enhancement model can outperform a general model when the test condition is matched to the training condition. Therefore, choosing the correct (matched) candidate model from a set of ensemble models is critical to achieve generalizability. Although the best decision criterion should be based directly on the evaluation metric, the need for a clean reference makes it impractical for employment. In this paper, we propose a novel specialized speech enhancement model selection (SSEMS) approach that applies a non-intrusive quality estimation model, termed Quality-Net, to solve this problem. Experimental results first confirm the effectiveness of the proposed SSEMS approach. Moreover, we observe that the correctness of Quality-Net in choosing the most suitable model increases as input noisy SNR increases, and thus the results of the proposed systems outperform another auto-encoder-based model selection and a general model, particularly under high SNR conditions. 

For more detail please check our <a href="https://www.isca-speech.org/archive_v0/Interspeech_2019/pdfs/2425.pdf" target="_blank">Paper</a>

### Installation ###

You can download our environmental setup at Environment Folder and use the following script.
```js
conda env create -f environment.yml
```
### Steps ###
1. We can use any clustering method to cluster the training data. In our implementation, we cluster the data based on the gender and SNR information. 
2. For developing the speech enhancement models, we can use any available architecture. In our setup, we use BLSTM to train the clustered training data.
3. For developing the Quality-Net model, please kindly refer to <a href="https://github.com/JasonSWFu/Quality-Net" target="Quality-Net">Quality-Net</a>
4. SSEMS_Main.py to select the best model based on Quality-Net

### Citation ###

Please kindly cite our paper, if you find this code is useful.

<a id="1"></a> 
R. E. Zezario, S.-W. Fu, X. Lu, H.-M. Wang, and Y. Tsao, “Specialized
Speech Enhancement Model Selection Based on Learned Non-Intrusive Quality Assessment Metric," in Proc. INTERSPEECH, pp.3168-
3172, 2019.
