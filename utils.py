import os, json
import config as c
from random import randint
import torch

def write_file(save_dir, text_data):
    with open(save_dir, "w") as f:
        for line in text_data:
            f.write(line)

def load_raw_file(file_path, mode = "read"):
    with open(file_path, "r") as f:
        if mode == "read":
            text_data = f.read() # since it's reading a raw text file, you will have to process this to remove \n etc
        elif mode == "readlines":
            text_data = f.readlines()
    return text_data


# will break if vocab file is not generated, fix this
vocab = load_raw_file(c.vocab_path)
stoi_mapping = {char : i for i, char in enumerate(vocab)}
itos_mapping = {i : char for i, char in enumerate(vocab)}

def stoi(text):
    encoded = []
    for char in text:
        if char in stoi_mapping:
            encoded.append(stoi_mapping[char])
    return encoded

def itos(self, tokenized_input):
    decoded = []
    for t in tokenized_input:
        if t in itos_mapping:
            decoded.append(itos_mapping[t])
    return "".join(decoded)


class WADataValidator:
    # WhatsAppDataValidator
    
    def __init__(self, mode = "train") -> None:
        # reads everything from config.py
        self.data_path = c.text_file_name
        self.mode = mode
        self.train_split= c.train_split
        assert self.train_split < 1.0
        self.test_split = 1 - self.train_split
        self.batch_size = c.batch_size
        self.block_size = c.block_size
        self.texts_to_ignore = ["<Media omitted>", "This message was deleted", "You deleted this message"]

        if os.path.isfile(self.data_path) and self.mode == "train":
            self.data = load_raw_file(self.data_path, mode="readlines")
            self.data = self.clean_data(self.data)
            # self.data = [char for line in self.data for char in line]

            _n = int(self.train_split * len(self.data))
            self.train_data, self.val_data = self.data[:_n], self.data[_n:]


        else:
            if self.mode == "train":
                raise FileNotFoundError(f"Data Path at {self.data_path} not found or missing !")
            # else:
            #     pass
        
        self.vocab_path = c.vocab_path
        self.train_data, self.val_data = torch.tensor(self.encode(self.train_data), dtype=torch.long), torch.tensor(self.encode(self.val_data), dtype=torch.long)

        if os.path.isfile(self.vocab_path):
            # load vocab file
            self.vocab = vocab
            # print(self.vocab)
            self.vocab_size = len(self.vocab)
            print("loaded vocab")

        else:
            # generate vocab and save it
            if self.mode != "train" and not os.path.isfile(self.data_path):
                raise FileNotFoundError(f"Cannot generate vocab.txt file since {self.data_path} does not exist. Change mode to train if you need to generate vocab")
            self.vocab, self.vocab_size = self.get_vocab(self.data)
            write_file(self.vocab_path, self.vocab)
            print("wrote")

        # self.stoi_mapping = {char : i for i, char in enumerate(self.vocab)}
        # self.itos_mapping = {i : char for i, char in enumerate(self.vocab)}
        
    def encode(self, string_input, as_tensor=False):
        encoded = []
        for char in string_input:
            if char in stoi_mapping:
                encoded.append(stoi_mapping[char])
        return encoded
    
    def encode_series(self, list_of_texts, as_tensor=False):
        encoded = []
        for text in list_of_texts:
            # print(text)
            encoded.append(self.encode(text))
        return encoded

    def decode(self, tokenized_input):
        decoded = []
        for t in tokenized_input:
            if t in itos_mapping:
                decoded.append(itos_mapping[t])
        return "".join(decoded)
        # return decoded
    
    def decode_series(self, list_of_tokens):
        decoded = []
        for tokens in list_of_tokens:
            decoded.append(self.decode(tokens))

        return decoded

    def clean_data(self, data):
        clean_data = []
        for t, text_line in enumerate(data):
            message = text_line.split("-")[-1].lstrip()
            if message in self.texts_to_ignore:
                continue

            for char in message:
                if ord(char) > 127:
                    message = message.replace(char, "")
            
            # next iteration, remove empty messages and make sure every text message sent is one line per author. Also make a Q and A type of thing
            clean_data.append(message)
        write_file("./datasets/combined_clean.txt", clean_data)
        clean_data = [char for line in clean_data for char in line]
        return clean_data

    def get_vocab(self, data):
        chars = []
        # since we're reading file in readlines and read
        for line in data:
            for char in line:
                if char not in chars:
                    chars.append(char)

        # chars = sorted(list(set(data)))
        chars = sorted(list(set(chars)))
        refined_chars = []
        for char in chars:
            if not ord(char) > 127: # refining so that we dont include non-english characters/emojis etc
                refined_chars.append(char) 

        return refined_chars, len(refined_chars)


    def get_batch(self, mode = "train"):
        x, y = [], []
        if mode == "train":
            sample_space = self.train_data
        elif mode == "val":
            sample_space = self.val_data
        else:
            raise Exception("Input args for 'mode' must be either 'train' or 'val'")

        for _ in range(self.batch_size):
            rand_pos = randint(0, len(sample_space) - self.block_size)
            x.append(sample_space[rand_pos: rand_pos + self.block_size])
            y.append(sample_space[rand_pos + 1 : rand_pos + self.block_size + 1])
        
        # x, y = torch.stack(x), torch.stack(y)
        return x, y



if __name__ == "__main__":
    config = WADataValidator()
    # print(config.encode("hi there"))
    # print(config.decode(config.encode("hi there")))

