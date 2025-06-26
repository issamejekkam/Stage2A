#to install "punkt"

# import nltk
# import ssl

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# nltk.download()

import numpy, tensorflow as tf, tensorflow_text, pandas, spacy
print("NumPy   :", numpy.__version__)
print("TF      :", tf.__version__)
print("TF-Text :", tensorflow_text.__version__)

print("spaCy:", pandaq.__version__)

