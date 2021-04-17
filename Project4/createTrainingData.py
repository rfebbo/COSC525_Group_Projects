import numpy as np



lines = ''
with open('beatles.txt') as f:
    for line in f:
        lines += line

lines = lines.replace('\n', '=')

sequences = []
stride = 3
window_size = 5
for c in range(0,len(lines),stride):
    #print(c)
    if(c + window_size < len(lines)):
        sequences.append(lines[c:c+window_size+1:1])
            
unique = np.unique(np.asarray(list(lines)))
def createInput(sequences):

    #print(unique)
    
    
    m = len(sequences)
    n = len(sequences[0])
    p = len(unique) # number of unique characters in data
    input = np.zeros((m,n,p))
    
    for s, seq in enumerate(sequences):
        for c, char in enumerate(seq):
            for u, un in enumerate(unique):
                if char == un:
                    input[s][c][u] = 1
                    
    return input
    
    
    
def createOutput(sequences):
    m = len(sequences)
    p = len(unique)
    
    output = np.zeros((m,p))
    for s, seq in enumerate(sequences[0:-1]):
        for u, un in enumerate(unique):
            if(sequences[s+1][0]) == un:
                output[s][u] = 1
    for u, un in enumerate(unique):
        if(sequences[0][0]) == un:
            output[-1][u] = 1
    print(output)
    return output

#createInput(sequences)
createOutput(sequences)

            

