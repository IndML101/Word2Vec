# Word2Vec training and Evaluation

The objective of this assignment is to learn distributed word representations that capture syntactic and semantic relation between them, using Skip-gram with negative sampling, published in [_Mikolov et al,_](https://browse.arxiv.org/pdf/1310.4546.pdf). The paper mentioned above reviews approaches used for the task at hand in the past  like **Continous bag of words**, **Skip-gram** and challenges associated with them. Where **Skip-gram** made the process of learning word representaions efficient, _Mikolov et. al._ suggested following extentions to Skip-gram model which further improved it around 2X-10X:

- Hierarchical softmax
- Negative sampling
- Noice contrastive estimation

Here in this notebook I have worked on **Skip-gram with negative sampling**. The project has following folder structure:
```
Word2Vec
├── Artifacts
│   ├── metadata
│   └── model
├── Data
│   ├── ds1.txt
│   └── ds2.txt
├── ds2_coding.pdf
├── __init__.py
├── main.py
├── Mikolov et al.pdf
├── notebooks
│   └── Word2Vec training and evaluation.ipynb
├── __pycache__
│   └── main.cpython-311.pyc
├── setup.py
├── src
│   ├── models.py
│   ├── __pycache__
│   └── utils.py
```

Root folder is "**Word2Vec**" which contains:

- **Artifacts:** for saving processed data, models, plots and other artifacts.
- **Data:** containing raw data
- **main.py:** main file, but its better to use **Word2Vec training and evaluation** file in the notebooks folder
- **src:** contains **utils.py** and **models.py** files with all the source code.
    - utils.py has two classes _DataIO_ for data read and write, _DataLoader_ for creating batches and negative samples.
    - models.py has three classes _SGNS_ - the model _per se_, _Word2Vec_ - model training wrapper and _EvaluateSGNS_ - to evaluate models.

### Objective

- **Skip-gram:**

$$
\underset{\theta}{\text{maximize}} \frac{1}{T} \sum_{t=1}^{T} \sum_{-c \le j \le c,j\ne0} \log p(w_{t+j} | w_t)
$$

$$p(w_O | w_I) = \frac{exp({v'_{w_O}}^T v_{w_I})}{\sum_{w=1}^{W} exp(v'_w v_{w_I})}$$

Skip-gram uses softmax to compute probability of words using softmax function which is very inefficient. Negative sampling reduces this computation by updating the objective funtion. 

- **Skip-gram with negative sampling:**

$$
\underset{\theta}{\text{maximize}} \sum_{(w,c) \in D} \log \sigma(v_c^Tv_w) + \sum_{(w,r) \in D'} \log \sigma(-v_r^Tv_w)
$$

$$\sigma(x) = \frac{1}{1+e^{-x}}$$

### Evaluation

For model evaluation I am just looking for closest word and manually check if it does make sense. There other ways like comparing model scores with bert scores for similar and dissimilar words.

**Sample output:**
```
{'When': ['doing.', 'bookshelf.', 'acorns', 'car,"', 'stumble'],
 'diamond.': ['are."', 'cobweb', 'side!"', 'day."', 'day:'],
 'disgusting!': ['word.', 'appreciated', 'Leave', 'Pinchy', 'hole!"'],
 'fireplace.': ['band.', 'brave', 'jewel', 'flown', 'teeth,'],
 'fruits': ['tree,', '"Thanks', 'paint.', 'shelf.', 'calling'],
 'insect.': ['Pandy.', 'secrets.', 'recommended', 'problems.Once', 'high!"'],
 'know!': ['elephant,', 'shadow.', 'back!', 'rough.', 'muffin.'],
 'lungs': ['snake!', 'needle', 'park.Once', 'white', '"Yuck!"'],
 'notes': ['ones,', 'Ash', 'tired.Once', 'organized', 'pumpkins'],
 'please!': ['part', 'mine!"', 'knot.', 'farm.Once', 'garden,']}
```

The results above shows top 5 most similar words to a given random word. The model for this was trained for 10000 steps for a vocab of 10000 words and embedding dimention of 300. The results don't look that good, will need to play around a bit with the parameters to tune it.