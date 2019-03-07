# UCF NLP Course; NLP Project 2019

## Classifying Label

#### Cultural Harassment	1 
#### Sexual Harassment		2
#### Personal Attacks		3

## Labeling is done 

#### Notes:
Some tweets were inexplicably split up in multiple lines. Folding them into one
line shifted Maxim's starting point. Maxim's first change now sits at line
2957 . Given that it doesn't look like he had to collapse any tweets, and his
first change occurred at line 2978 prior to me collapsing tweets, that suggests
a current shift of -21 lines. So I assume Maxim's block starts at what is now
line 2919.

## Plan of Attack & Dates:
#### Data collection 					Feb 10  	[V]
#### Data labeling into 3 categories	Feb 17		[V]

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
    ...
    (CAP6640_PROJ)$ python3 devtools/dataset_stats.py

#### Project proposal					End of Feb 	[V]

The proposal, as submitted, is included in the project as the `proposal.tex`
and `proposal.bib` files. Typesetting with LaTeX requires the files
`acmcopyright.sty` and `sig-alternate-05-2015.cls`.

#### Text tokenization					Mar 5 		[V]

The *NLTK* library has been added to the list of requirements for the project.

#### Data addition						Mar 10 		[]
	Some "great" resources:
		https://www.reddit.com/r/The_Donald/hot/	- cultural (it is hard to find cultural attacks in general, people tend to be very sensitive to these, so they are most likely banned)
		https://www.reddit.com/r/RoastMe/			- goldmine for personal attacks

		Below are sexual sources, please be cautios when opening, these are not safe to watch in public.
		https://www.reddit.com/r/dirtypenpals/top/?t=all
		https://www.reddit.com/r/gonewildaudio		


#### Text processing					Mar 10 		[]
#### Classifier							Mar 15 		[]
	SVM
	BAYES
	LSTM
	(?) cnn
	(?) n-gram
	(?) rnn
#### Hyperparameter tuning				Mar 22 		[]
#### Presentation						-- 			[]
#### Final report 						(?) 		[]
