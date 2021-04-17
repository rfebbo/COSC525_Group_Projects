def create_sequences(window_size, stride):
    lines = ''
    with open('beatles.txt') as f:
        for line in f:
            lines += line

    lines = lines.replace('\n', '=')

    sequences = []
    for c in range(0,len(lines),stride):
        if(c + window_size < len(lines)):
            sequences.append(lines[c:c+window_size+1:1])

    with open(f'sequences{stride}_{window_size}','w') as f:
        for line in sequences:
            f.write(line + '\n')