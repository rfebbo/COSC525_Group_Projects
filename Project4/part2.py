from numpy import zeros
from part1 import create_sequences


def createInputOutput(uni, sequences):
    m = len(sequences)
    n = len(sequences[0]) - 1
    p = len(uni)
    input = zeros((m,n,p))
    
    for s, seq in enumerate(sequences):
        for c, char in enumerate(seq[0:-1]):
            for u, un in enumerate(uni):
                if char == un:
                    input[s][c][u] = 1
    
    output = zeros((m,p))
    for s, seq in enumerate(sequences):
        for u, un in enumerate(uni):
            if(sequences[s][-1]) == un:
                output[s][u] = 1
    
    # for s, seq in enumerate(sequences[0:-1]):
    #     for u, un in enumerate(uni):
    #         if(sequences[s+1][0]) == un:
    #             output[s][u] = 1
    # for u, un in enumerate(uni):
    #     if(sequences[0][0]) == un:
    #         output[-1][u] = 1
    return input, output


# u, s = create_sequences(5,3)
# i, o = createInputOutput(u, s)
