# Generative-based-KD

This is the official repo of article [*Generative Model-based Feature Knowledge Distillation for Action Recognition*](https://ojs.aaai.org/index.php/AAAI/article/view/29473) which is accepted by AAAI-24.

For action detection task(TAL), we follow the environment setup of AFSD, which can be found [here](https://github.com/TencentYoutuResearch/ActionDetection-AFSD).

For action recognition task, we use I3D and Top-I3D model for teacher and student model, respectively. we simply add an classification head after the backbone.

If you have any question about your environment setup, please first visit [AFSD issue area](https://github.com/TencentYoutuResearch/ActionDetection-AFSD/issues) for solutions, then feel free to ask in this repo issue area, we will offer suggestions ASAP. 
