from numpy import unique, asarray

def create_sequences(file, window_size, stride):
    lines = ''
    with open(file) as f:
        for line in f:
            lines += line

    lines = lines.replace('\n', '=')
    uni = unique(asarray(list(lines)))
    sequences = []
    stride = stride
    window_size = window_size
    for c in range(0,len(lines),stride):
        if(c + window_size < len(lines)):
            sequences.append(lines[c:c+window_size+1:1])

    with open(f'sequences{window_size}_{stride}','w') as f:
        for line in sequences:
            f.write(line + '\n')

    return uni, sequences
