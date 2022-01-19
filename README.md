# Deep-Learning
## AI subjects

### Artificial Intelligence Applications: Autonomous Vehicles (Cool , Like waymo)

![image](https://user-images.githubusercontent.com/71330579/149942966-8a88ff9e-900b-4090-875e-ba43140a39ad.png)


### Ethical decisions sur les Applications des voitures autonomes ? (bizzare , safety procedures)

![image](https://user-images.githubusercontent.com/71330579/149942828-f307ecb9-7cf7-4c62-9e54-f3c2c5352018.png)

### Réseau neurones sites :

* https://www.cs.ryerson.ca/~aharley/vis/conv/

* http://neuralnetworksanddeeplearning.com/chap1.html

* http://cs231n.stanford.edu

* https://www.whichfaceisreal.com

* Trading bots : https://www.youtube.com/watch?v=vT1JzLTH4G4&list=PLf7L7Kg8_FNxHATtLwDceyh72QQL9pvpQ

Class deep learning links : 

* https://course.fast.ai/

* https://colab.research.google.com/github/fastai/fastbook/blob/master/01_intro.ipynb#scrollTo=I8AlZFww4f9y

* https://colab.research.google.com/github/fastai/fastbook/blob/master/02_production.ipynb#scrollTo=wvNP5Pj26Bee

* https://www.fast.ai/

* https://github.com/fastai/fastbook

* Open source INSA Toulouse machine learning : https://github.com/wikistat/AI-Frameworks


# Application of Artificial Intelligence (AI) in Prosthetic and Orthotic Rehabilitation #

Source pdf : 
[research_prostethic.pdf](https://github.com/Mohamed-Khalil67/Deep-Learning/files/7890650/research_prostethic.pdf)


### Note book `Intro` : 

- Notes écrit sur deux lessons de machine learning de https://course.fast.ai/ .

### Note Book `Atelier-Keras` :

- Notes écrti sur le gitlab du benjamin dellard
- Original source : https://github.com/wikistat/AI-Frameworks
- Benjamin git : https://gist.github.com/bdallard/ed166ad884491c191d877c07e0b18008

#### Library documentation :

* Keras : https://keras.io/

![image](https://user-images.githubusercontent.com/71330579/150099895-a85c7f2f-ada8-4070-9c20-35be02bee55a.png)


`Keras applications` :

![image](https://user-images.githubusercontent.com/71330579/150116052-90491b6a-2718-442c-9665-da57d5085176.png)

Features avec VGG16 :- 

```
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

model = VGG16(weights='imagenet', include_top=False)

img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

features = model.predict(x)
```
