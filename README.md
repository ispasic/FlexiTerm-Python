## [FlexiTerm](http://users.cs.cf.ac.uk/I.Spasic/flexiterm/): a software tool to automatically recognise multi-word terms in text documents.

FlexiTerm takes as input a corpus of ASCII documents and outputs a ranked list of automatically recognised multi-word terms.

If you use FlexiTerm in your work/research, please cite the following papers:

[Spasić, I., Greenwood, M., Preece, A., Francis, N., & Elwyn, G. (2013). FlexiTerm: a flexible term recognition method. Journal of biomedical semantics, 4(1), 27.](https://jbiomedsem.biomedcentral.com/articles/10.1186/2041-1480-4-27)

For more information, please visit the [FlexiTerm](http://users.cs.cf.ac.uk/I.Spasic/flexiterm/) website.

Python requirements:

Python 		version "3.7.4"

Dependencies:

jellyfish 	version "0.8.2"
nltk 		version "3.4.5"
numpy 		version "1.20.3"
spacy 		version "3.0.6"

Folders:

config : System configuration files.
out    : Output files.
text   : Input files (plain text only).

Files:

flexiterm.py          : The main python file.
flexiterm.ipynb       : Jupyter notebook version of flexiterm.py.
flexiterm.sqlite      : An sqlite database used by flexiterm.py.
out/terminology.csv   : A table of results: id | variant | c | f | df | c_idf
out/terminology.html  : A table of results: Term ID | Termhood | Term variant | Term variant frequency
out/concordances.html : Concordances of terms listed in terminology.html.
out/corpus.html       : Input text annotated with occurrences of terms listed in terminology.html.
out/annotations.json  : Annotations of term occurrences in the input files using the spaCy format for 
                        training data: https://spacy.io/usage/training#training-data
                        They can be used for visualisation or downstream processing by other applications.
config/settings.txt   : Specifies:
                        * pattern  : term formation pattern(s)
                        * stoplist : the location of the stoplist
                        * Smin     : Jaro-Winkler similarity threshold
                        * Amin     : minimum (implicit) acronym frequency
                        * Fmin     : minimum term candidate frequency
                        * Cmin     : minimum C-value
                        * acronyms : acronym recognition mode (implicit or explicit)

                        Default settings:
                        * pattern  : "(((((NN|JJ) )*NN) IN (((NN|JJ) )*NN))|((NN|JJ )*NN POS (NN|JJ )*NN))|(((NN|JJ) )+NN( CD)?)"
                        * stoplist : ./config/stoplist.txt
                        * Smin     : 0.962
                        * Amin     : 5
                        * Fmin     : 2
                        * Cmin     : 1
                        * acronyms : explicit
config/stoplist.txt   : A list of stopwords.
config/schema.sql     : A schema of the database stored in flexiterm.sqlite.
                        
                        
FlexiTerm takes as input a corpus of ASCII documents and outputs 
a ranked list of automatically recognised multi-word terms.

To run FlexiTerm:

1. Place input files (plain text only) into a folder named "text".

2. OPTIONAL: Replace file config/stoplist.txt with your own if needed.

3. Execute flexiterm.py from the command line: python flexiterm.py
   OR run the following Jupyter notebook: flexiterm.ipynb

4. Check the results by double-clicking out/terminology.html from which 
   you can navigate to out/concordances.html and then to out/corpus.html.
