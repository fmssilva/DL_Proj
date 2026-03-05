import os

def convert_to_txt(filename):
    # Get the base name without extension
    base_name = os.path.splitext(filename)[0]
    txt_filename = f"{base_name}.txt"

    # Read the contents of the input file
    with open(filename, 'r', encoding='utf-8') as infile:
        content = infile.read()

    # Write the contents to the new .txt file
    with open(txt_filename, 'w', encoding='utf-8') as outfile:
        outfile.write(content)

    return txt_filename

convert_to_txt('task1.ipynb')