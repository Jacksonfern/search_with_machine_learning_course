import fasttext
import argparse

parser = argparse.ArgumentParser(description="Generate synonyms for top words.")
general = parser.add_argument_group("general")
general.add_argument("--model", required=True, help="Path to fasttext model")
general.add_argument("--input", required=True, help="Path to list of words")
general.add_argument("--output", required=True, help="Path to output synonym list")

args = parser.parse_args()

model_path = args.model
words_path = args.input
output_path = args.output

model = fasttext.load_model(model_path)

with open(words_path, 'r') as w, open(output_path, 'w') as o:
    for line in w:
        synonyms = [line.strip()]
        nns = model.get_nearest_neighbors(line.strip())
        for (similarity, word) in nns:
            if similarity >= 0.8:
                synonyms.append(word)
        if len(synonyms) > 1:
            o.write(",".join(synonyms) + '\n')