
#######################
#
# Pytorch stuff
#

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# local PC
path = "./"

# Production parameters
batch_size = 64
TEXT_PORTION_SIZE = 200
NUM_ITER = 1000
LEARNING_RATE = 0.005

HIDDEN_SIZE = 512
EMBEDDING_SIZE = 10

decode = 'abcdefghijklmnopqrstuvwxyz ,;.:?!'
DEVICE = "cpu"

class CharLSTM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size):
        super(CharLSTM, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = 2

        self.embedding = nn.Embedding(input_size, embedding_size)
        # LSTM non richiede la lunghezza della sequenza
        self.rnn = nn.LSTM(input_size=self.embedding_size,
                           hidden_size=self.hidden_size,
                           num_layers=self.num_layers,
                           batch_first=False
                          )
        self.fc = nn.Linear(hidden_size, output_size)

    # outputs the logits
    # expects inputs as size [seq_len,batch_size]
    def forward(self, inputs, hidden_and_cell=None):

        if hidden_and_cell is None:
            hidden_and_cell = self.init_zero_state(inputs.size(1))
        
        embedded = self.embedding(inputs)
        
        output, hidden_and_cell = self.rnn(embedded, hidden_and_cell)
        # 1. output dim: [sentence length, batch size, hidden size]
        # 2. hidden dim: [num layers, batch size, hidden size]
        # 3. cell dim: [num layers, batch size, hidden size]

        prediction = self.fc(output)
        # prediction dim: [sentence length, batch size, output size]
        
        return prediction, hidden_and_cell

    def init_zero_state(self, batch_size=1):
        init_hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
        init_cell = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=DEVICE)
        return (init_hidden, init_cell)

validation = CharLSTM(input_size=len(decode),embedding_size=EMBEDDING_SIZE,hidden_size=HIDDEN_SIZE,output_size=len(decode))
validation = validation.to(DEVICE)
validation.load_state_dict(torch.load(path+'lucrezio04.pth',map_location=torch.device(DEVICE)))
validation.eval()

def evaluate(model, prime_str='a', predict_len=100, temperature=0.8):
    ## based on https://github.com/spro/practical-pytorch/
    ## blob/master/char-rnn-generation/char-rnn-generation.ipynb

    prime_input = torch.tensor([decode.index(c) for c in prime_str]).long()
    prime_input = prime_input.view(1,1,-1).contiguous().to(DEVICE)
    predicted = prime_str

    hidden_and_cell = model.init_zero_state(1)

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden_and_cell = model(prime_input[:,:,p], hidden_and_cell)
    inp = prime_input[:,:,-1]

    # predict text
    for p in range(predict_len):

        outputs, hidden_and_cell = model(inp, hidden_and_cell)

        # Sample from the network as a multinomial distribution
        output_dist = outputs.data.view(-1).div(temperature).exp() # e^{logits / T}
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string
        predicted_char = decode[top_i]
        predicted += predicted_char

        # use as next input
        inp = torch.tensor([[decode.index(predicted_char)]]).long()
        inp = inp.to(DEVICE)

    return predicted


def createText(text,nChar):
	res = evaluate(validation, text, nChar)
	return res


#######################
# https://www.digitalocean.com/community/tutorials/how-to-use-templates-in-a-flask-application

from flask import *
app = Flask(__name__)

def checkText(text):
	msg = ""
	decode = 'abcdefghijklmnopqrstuvwxyz ,;.:?!'

	if len(text) == 0:
		msg = "Si inserisca un testo iniziale"
	if len(text) > 20:
		msg = "Si inseriscano al massimo 20 caratteri iniziali"
	if msg == "":
		for c in text:
			if c not in decode:
				msg = "Si usino solo i seguenti caratteri: abcdefghijklmnopqrstuvwxyz ,;.:?!"
				break
	return msg


def checkNum(nChar,text):
	msg = ""
	if nChar.isdigit():
		nChar = int(nChar)
		if nChar <= len(text):
			msg = "Si richieda un numero di caratteri maggiore della lunghezza del testo iniziale"
		if nChar > 500:
			msg = "Si richiedano al massimo 500 caratteri"
	else:
		msg = "Si richieda un numero di caratteri da creare"
	return msg,nChar

@app.route("/", methods=["GET", "POST"])
def home():

	out1 = ""
	out2 = ""

	if request.method == "POST":
		text = request.form["text"]
		nChar = request.form["nChar"]
		
		msg1 = checkText(text)
		msg2,nChar = checkNum(nChar,text)

		if msg1 == "" and msg2 == "":
			out1 = "Testo creato:"
			out2 = createText(text,nChar)
		else:
			out1 = msg1
			out2 = msg2

	return render_template("home.html",output1=out1,output2=out2)

#######################

if __name__ == "__main__":
	app.run()
