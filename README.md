# Deep Reinforcement Learning based Evasion Generative Adversarial Network for Botnet Detection

Botnet detectors based on machine learning are potential targets for adversarial evasion attacks. Several research works employ adversarial training with samples generated from generative adversarial nets (GANs) to make the botnet detectors adept at recognising adversarial evasions. However, the synthetic evasions may not follow the original semantics of the input samples. This paper proposes a novel GAN model leveraged with deep reinforcement learning (DRL) to explore semantic aware samples and simultaneously harden its detection. A DRL agent is used to attack the discriminator of the GAN that acts as a botnet detector. The discriminator is trained on the crafted perturbations by the agent during the GAN training, which helps the GAN generator converge earlier than the case without DRL. We name this model RELEVAGAN, i.e. ["relieve a GAN" or deep REinforcement Learning-based Evasion Generative Adversarial Network] because, with the help of DRL, it minimises the GAN's job by letting its generator explore the evasion samples within the semantic limits. During the GAN training, the attacks are conducted to adjust the discriminator weights for learning crafted perturbations by the agent. RELEVAGAN does not require adversarial training for the ML classifiers since it can act as an adversarial semantic-aware botnet detection model.
![](RELEVAGAN.svg "RELEVAGAN Architecture")


## Prerequisites
* Tensorflow
* Keras
* Numpy
* For the rest of the packages please refer to header.py file inside the project directory.


## Cite this Work
```
Randhawa, Rizwan Hamid, et al. "Deep Reinforcement Learning based Evasion Generative Adversarial Network for Botnet Detection." arXiv preprint arXiv:2210.02840 (2022).
```
### Bibtex
```
@misc{https://doi.org/10.48550/arxiv.2210.02840,
  doi = {10.48550/ARXIV.2210.02840},
  
  url = {https://arxiv.org/abs/2210.02840},
  
  author = {Randhawa, Rizwan Hamid and Aslam, Nauman and Alauthman, Mohammad and Khalid, Muhammad and Rafiq, Husnain},
  
  keywords = {Cryptography and Security (cs.CR), Artificial Intelligence (cs.AI), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Deep Reinforcement Learning based Evasion Generative Adversarial Network for Botnet Detection},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}


```