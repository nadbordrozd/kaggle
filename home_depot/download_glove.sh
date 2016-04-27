#curl http://nlp.stanford.edu/data/glove.6B.zip > data/glove_6B.zip
#curl http://nlp.stanford.edu/data/glove.42B.300d.zip > data/glove_42B_300d.zip

curl http://nlp.stanford.edu/data/glove.840B.300d.zip > data/glove_840B_300d.zip

unzip data/glove_840B_300d.zip -d data/
rm data/glove_840B_300d.zip
