import argparse
import multiprocessing
import glob
from tqdm import tqdm
import os
import xml.etree.ElementTree as ET
from pathlib import Path
import re

def transform_name(product_name):
    # strip space characters at the beginning and at the end of the text 
    transformed_name = product_name.strip().lower()
    # remove all non non-alphanumeric characters other than underscore
    transformed_name = re.sub(r'[^A-Za-z0-9_\-\s]', '', transformed_name)
    # trim excess space characters 
    transformed_name = re.sub(r'\s+', ' ', transformed_name)

    return transformed_name

# Directory for product data
directory = r'/workspace/datasets/product_data/products/'
categories_filename = r'/workspace/datasets/product_data/categories/categories_0001_abcat0010000_to_pcmcat99300050000.xml'

parser = argparse.ArgumentParser(description='Process some integers.')
general = parser.add_argument_group("general")
general.add_argument("--input", default=directory,  help="The directory containing product data")
general.add_argument("--output", default="/workspace/datasets/fasttext/output.fasttext", help="the file to output to")
general.add_argument("--label", default="id", choices=['id', 'name'], help="id is default and needed for downsteam use, but name is helpful for debugging")

# Setting min_products removes infrequent categories and makes the classifier's task easier.
general.add_argument("--min_products", default=0, type=int, help="The minimum number of products per category (default is 0).")

args = parser.parse_args()
output_file = args.output
path = Path(output_file)
output_dir = path.parent
if os.path.isdir(output_dir) == False:
        os.mkdir(output_dir)

if args.input:
    directory = args.input
min_products = args.min_products
names_as_labels = False
if args.label == 'name':
    names_as_labels = True

def _label_filename(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    labels = []
    for child in root:
        # Check to make sure category name is valid and not in music or movies
        if (child.find('name') is not None and child.find('name').text is not None and
            child.find('categoryPath') is not None and len(child.find('categoryPath')) > 0 and
            child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text is not None and
            child.find('categoryPath')[0][0].text == 'cat00000' and
            child.find('categoryPath')[1][0].text != 'abcat0600000'):
              # Choose last element in categoryPath as the leaf categoryId or name
              if names_as_labels:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][1].text.replace(' ', '_')
              else:
                  cat = child.find('categoryPath')[len(child.find('categoryPath')) - 1][0].text
              # Replace newline chars with spaces so fastText doesn't complain
              name = child.find('name').text.replace('\n', ' ')
              labels.append((cat, transform_name(name)))
    return labels

def get_ancestors_category(categories_filename, label):
    tree = ET.parse(categories_filename)
    root = tree.getroot()
    cats_anc = {}

    for child in root:
        cur_cat = child.find(label).text
        cats_anc[cur_cat] = cur_cat
        deep = 0

        cat_path = child.find('path')
        for cat in cat_path:
            cat_id = cat.find(label).text
            deep += 1
            if deep == 2:
                cats_anc[cur_cat] = cat_id
                break
    return cats_anc

def filter_min_products(all_labels, min_freq, categories_filename, label):
    cat_ancs = get_ancestors_category(categories_filename, label)
    cats_count = {}
    for label_list in all_labels:
        for (cat, _) in label_list:
            anc_cat = cat_ancs.get(cat, cat)
            if anc_cat in cats_count:
                cats_count[anc_cat] += 1
            else:
                cats_count[anc_cat] = 1
    
    filtered_labels = []
    for label_list in all_labels:
        for (cat, name) in label_list:
            anc_cat = cat_ancs.get(cat, cat)
            if cats_count[anc_cat] >= min_freq:
                filtered_labels.append((anc_cat, name))

    return filtered_labels

if __name__ == '__main__':
    files = glob.glob(f'{directory}/*.xml')
    print("Writing results to %s" % output_file)
    with multiprocessing.Pool() as p:
        all_labels = tqdm(p.imap(_label_filename, files), total=len(files))
        filtered_labels = filter_min_products(list(all_labels), min_products, categories_filename, args.label)
        with open(output_file, 'w') as output:
            for (cat, name) in filtered_labels:
                output.write(f'__label__{cat} {name}\n')
