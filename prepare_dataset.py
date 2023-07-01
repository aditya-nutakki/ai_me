import os
import sys
import config as c
from utils import write_file

def main():
    dataset_path = os.path.dirname(c.text_file_name)
    os.makedirs(dataset_path, exist_ok=True)
    conversations_path = sys.argv[1] # D:\work\datasets\circles\processed

    converstations = os.listdir(conversations_path)

    combined = []
    for conversation in converstations:
        if conversation.endswith(".txt"):
            converstation_path = os.path.join(conversations_path, conversation)
            with open(converstation_path, "r") as f:
                conversation = f.readlines()
                combined.extend(conversation)

    write_file(c.text_file_name, combined)
    print(f"saved combined text dataset to {c.text_file_name} !")


if __name__ == "__main__":
    main()