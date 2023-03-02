import sys
import argparse
import os

height = 0
width = 0
channels = 0

def image_converter(csv_fname, im_format):
    if im_format == 'mnist':
        height = 28
        width = 28
        channels = 1
    elif im_format == 'cifar-10':
        height = 32
        width = 32
        channels = 3
    else: # imagenet
        height = 224
        width = 224
        channels = 3

    try:
        csv_file = open(csv_fname, 'r')
        contents = csv_file.readlines()
        contents = contents[0].strip('\n').split(',')
        csv_file.close()
    except:
        "Error opening image CSV file."
        sys.exit(0)

    # extract label
    label = contents[0]
    output_str = label + ',' + str(height) + ',' + str(width) + ',' + str(channels) + ','
    for i in range(1,len(contents), 1):
        output_str += (contents[i] + ',')

    # save to file
    out_fname = 'image.ptxt'
    out_file = open(out_fname, 'w')
    out_file.write(output_str[:len(output_str)-1])
    out_file.close()

def file_path(filepath):
    if os.path.isfile(filepath) or os.path.isdir(filepath):
        return filepath
    else:
        print(filepath + ' is not a file or directory')
        exit(-1)

def main():
    parser = argparse.ArgumentParser(description='Format Converter for REDsec')
    parser.add_argument('--format', help='MNIST, CIFAR-10, or ImageNet', type=str.lower, choices=['mnist', 'cifar-10', 'imagenet'], required=True)
    parser.add_argument('--image', help='Path to input image', type=file_path, required=True)

    arguments = parser.parse_args()

    image_converter(arguments.image, arguments.format)

main()