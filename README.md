# UCF NLP Course
NLP Project 2019


## Classifying Label

#### Cultural Harassment - 1 
#### Sexual Harassment - 2
#### Personal Attacks - 3

#### Rolando will take first 1/3 of data - 1-2939
Finished labeling


Some tweets were inexplicably split up in multiple lines. Folding them into one
line shifted Maxim's starting point. Maxim's first change now sits at line
2957 . Given that it doesn't look like he had to collapse any tweets, and his
first change occurred at line 2978 prior to me collapsing tweets, that suggests
a current shift of -21 lines. So I assume Maxim's block starts at what is now
line 2919.

#### Maxim will take 2nd 1/3 of data     - 2940 - 5879
#### Marc will take 3rd 1/3 of data      - 5880 - 8818

## Plan of Attack & Dates:
#### Data collection 			Feb 10  [check]

#### Data labeling into 3 categories	Feb 17 []

Data label distribution as of 28 February 2019

| Label | Raw Count | Percentage |
| :---  |      ---: |       ---: |
| Non-Bullying | 7,649 | 86.8% |
| Cultural | 559 | 6.3% |
| Sexual | 165 | 1.9% |
| Personal | 444 | 5.0% |

Data produced using the `dataset_stats.py` script in the `devtools` directory.
The Python program was ran under a 'virtualenv' built using
the requirements present in the repository's `requirements.txt` file:

    $ python3 -m virtualenv -p $(which python3) --always-copy ~/pythonenv/CAP6640_PROJ
    ...
    $ . ~/pythonenv/CAP6640_PROJ/bin/activate
    (CAP6640_PROJ)$ python3 -m pip install -r requirements.txt

#### Project proposal			[]
#### Text tokenization			Feb 24 []
#### Text processing			Mar 1 []
#### Classifier				Mar 15 []
#### Hyperparameter tuning		Mar 22 []
#### Presentation				-- []
#### Final report (?) []
