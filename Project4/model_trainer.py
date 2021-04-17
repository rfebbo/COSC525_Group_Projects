from models import *
from part1 import create_sequences
from part2 import createInputOutput


uni, seq = create_sequences(3,5)
x_train, y_train = createInputOutput(uni,seq)


model = simple_model1()
model.summary()

# model.fit(x_train, y_train, epochs=10)
