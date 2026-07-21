import sys
sys.path.append('./modules')

import pickle
from BPE_tokenizer import BPETokenizer, BPEEncoder
from Model import TransformerEncoder, TransformerDecoder, Seq2Seq
from Predict import PredictionModule

with open("../data/predicter/predicter_1.pkl", 'rb') as f:
    predicter = pickle.load(f)

print(f"Eng-Pol Translator ready. Type 'quit' to exit.\n")

while True:
    snt_eng = input("EN: ")
    if snt_eng.lower() == 'quit':
        break
    print(f"PL: {predicter.translate_snt(snt_eng)}\n")
    
    