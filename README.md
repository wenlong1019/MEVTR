# MEVTR

MEVTR is a Multilingual model Enhanced with Visual Text Representations, which complements text representations and extends the multilingual representation space with visual text representations. First, the visual encoder focuses on the glyphs and structure of the text to obtain visual text representations, while the textual encoder obtains textual representations. Then, multilingual representations are enhanced by aligning and fusing visual text representations and textual representations.

# Setup

You need to configure the appropriate datasets, models and training environments according to the README.md file in each directory of the file.

# Train

###### NER TASK

```
python src/train_ner.py
```

###### POS TASK

```
python src/train_pos.py
```
