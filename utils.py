import os, json
import config as c

def write_file(save_dir, text_data):
    with open(save_dir, "w") as f:
        for line in text_data:
            f.write(line)

def load_raw_file(file_path):
    with open(file_path, "r") as f:
        # text_data = f.readlines() # since it's reading a raw text file, you will have to process this to remove \n etc
        text_data = f.read() # since it's reading a raw text file, you will have to process this to remove \n etc
    return text_data

def stoi(text):
    return 

def itos():
    return

class ConfigureData:
    def __init__(self) -> None:
        # reads everything from config.py
        self.data_path = c.text_file_name
        
        if os.path.isfile(self.data_path):
            self.data = load_raw_file(self.data_path)
        else:
            raise FileNotFoundError(f"Data Path at {self.data_path} not found or missing !")
        
        self.vocab_path = c.vocab_path

        if os.path.isfile(self.vocab_path):
            # load vocab file
            self.vocab = load_raw_file(self.vocab_path)
            # print(self.vocab)
            print("loaded vocab")

        else:
            # generate vocab and save it
            self.vocab, self.vocab_size = self.get_vocab(self.data)
            write_file(self.vocab_path, self.vocab)
            print("wrote")


    def get_vocab(self, data):
        chars = sorted(list(set(data)))
        refined_chars = []
        for char in chars:
            if not ord(char) > 127: # refining so that we dont include non-english characters/emojis etc
                refined_chars.append(char) 

        return refined_chars, len(chars)
    

if __name__ == "__main__":
    config = ConfigureData()

