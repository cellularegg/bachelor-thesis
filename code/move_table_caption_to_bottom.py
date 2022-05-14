import glob

# Depending from which directory you run this script get all paths of the tables
file_list = glob.glob('../tables/**/*.tex')
file_list = file_list + glob.glob('./tables/**/*.tex')
file_list = file_list + glob.glob('./tables/*.tex')

for table_path in file_list:
    print(f'Processing {table_path}')
    with open(table_path, 'r') as f:
        lines = f.readlines()
    # Get the index of line with caption in it
    caption_idx = -1
    for i in range(len(lines)):
        if '\\caption' in lines[i]:
            caption_idx = i
            break
    # Remove caption and insert it one line before the end
    lines.insert(-1, lines.pop(caption_idx))

    # Get the index of the line with the label in it
    label_idx = -1
    for i in range(len(lines)):
        if '\\label' in lines[i]:
            label_idx = i
            break
    # Remove label and insert it one line before the end
    lines.insert(-1, lines.pop(label_idx))

    # Write back to file
    with open(table_path, 'w') as f:
        f.writelines(lines)
