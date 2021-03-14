import numpy as np
# N 0
# C 1
# H 2
# W 3
def convolve_2d(x, f, b, stride, padding):

    # shorthand for inputshape sizes
    Ni = x.shape[0]
    Ci = x.shape[1]
    Hi = x.shape[2]
    Wi = x.shape[3]

    # shorthand for filter shape sizes
    Nf = f.shape[0]
    Cf = f.shape[1]
    Hf = f.shape[2]
    Wf = f.shape[3]

    # determine output shape size
    No = Ni
    Co = Nf
    Ho = int((Hi - Hf + padding * 2 + stride)  / stride)
    Wo = int((Wi - Wf + padding * 2 + stride) / stride)
    
    if (Cf != Ci):
        print(f'conv2d: number of channels in filter ({Cf}) does not match input ({Ci})')
        exit()

    y = np.empty((No,Co,Ho,Wo))

    for n_o in range(No):
        for c_o in range(Co):
            for h_o in range(Ho):
                for w_o in range(Wo):
                    val = 0

                    for c_f in range(Cf):
                        for h_f in range(Hf):
                            for w_f in range(Wf):
                                h_i = h_o * stride - padding + h_f 
                                w_i = w_o * stride - padding + w_f

                                if (h_i < Hi and w_i < Wi and h_i >= 0 and w_i >= 0):
                                    val += x[n_o, c_f, h_i, w_i] * f[c_o, c_f, h_f, w_f]
                    
                    y[n_o, c_o, h_o, w_o] = val + b[c_o]

    return y

test=False
if (test):
    x = [  [[1,1,0,0,2],
            [1,0,1,1,0],
            [1,0,0,1,0],
            [1,0,2,2,0],
            [1,0,2,1,1]],
        [[2,1,0,0,2],
            [2,1,1,2,2],
            [0,1,1,0,1],
            [2,2,1,0,2],
            [0,0,2,0,0]],
        [[2,2,2,0,0],
            [1,2,0,1,1],
            [2,1,2,0,2],
            [0,0,0,2,2],
            [2,2,1,1,1]]]

    f = [   [[-1, 0, 1],
            [-1, 0,-1],
            [-1,-1,-1]],
            [[-1, 0,-1],
            [ 1, 1, 1],
            [-1, 0,-1]],
            [[ 1, 1,-1],
            [-1,-1, 1],
            [ 0, 0, 1]]]

    x = np.asarray(x)
    f = np.asarray(f)

    # print(x.shape)
    # print(f.shape)

    x = np.reshape(x, (1,3,5,5))
    f = np.reshape(f, (1,3,3,3))
    # print(x)
    # print(f)

    # print(x.shape)
    # print(f.shape)
    y = convolve_2d(x, f, [0], 2, 1)

    print(y)
