import torch
import numpy as np
import sys
from tqdm.auto import tqdm

def encode_rnn_params(num_layers,batch_size,num_hidden,seq_length,epochs,N):
  """
  This helper function returns the RNN hyperparameters as a dictionary. This is useful for 
  passing the parameters into a training loop for creating the hidden state.
  Hyperparameters:
  num_layers -> number of layers in the rnn
  batch_size -> number of sequence samples that will go into a training batch
  num_hidden -> number of hidden units of the layers
  epochs -> number of training epochs to train the model for
  seq_length -> the number of timesteps that will be fed into the model each timestep
  N -> total size of the training sequence
  """
  return {
      "num_layers":num_layers,
      "batch_size":batch_size,
      "num_hidden":num_hidden,
      "seq_length":seq_length,
      "N":N,
      "num_epochs":epochs
  }

def rnn_standard_training_loop(net,loss_fn,optimizer,device,net_hyperparams):
  
  # retrieve training hyperparameters
  epochs = net_hyperparams["num_epochs"]
  num_layers = net_hyperparams["num_layers"]
  batch_size = net_hyperparams["batch_size"]
  num_hidden = net_hyperparams["num_hidden"]
  seq_length = net_hyperparams["seq_length"]
  N = net_hyperparams["N"]

  # initialize the losses
  losses = np.zeros(epochs)

  # loop over epochs
  for epoch in tqdm(range(epochs)):
    # loop over data segements
    seg_losses = list()
    # reset the hidden state
    hidden_state = torch.zeros(num_layers,batch_size,num_hidden).to(device)

    for time in range(N-seq_length):
      # fetch a snippet of data
      X = data[time:time+seq_length].view(seq_length,1,1)
      y = data[time+seq_length].view(1,1)

      # forward pass 
      y_hat, hidden_state = net(X)
      final_value = y_hat[-1]

      # calculate the loss
      loss = loss_fn(final_value,y)

      # turn off grads
      optimizer.zero_grad()

      # back propagation
      loss.backward()
      optimizer.step()

      # loss from this segment
      seg_losses.append(loss.item())

    
    # average losses from this epoch
    losses[epoch] = np.mean(seg_losses)
    msg = f"EPOCH: {epoch + 1}/{epochs} | LOSS: {losses[epoch]}"
    sys.stdout.write("\r" + msg)
  return losses,y_hat

def rnn_standard_testing_loop(net,data,net_hyperparams):
  # retrieve hyperparameters
  seq_length = net_hyperparams["seq_length"]
  N = net_hyperparams["N"]
  num_hidden = net_hyperparams["num_hidden"]

  # initialize a new hidden state
  h = np.zeros((N,num_hidden))

  # initialize predicted values
  y_hat = np.zeros(N)
  y_hat[:] = np.nan

  net.eval()

  # testing loop
  for time in tqdm(range(N-seq_length)):
    X = data[time:time+seq_length].view(seq_length,1,1)

    with torch.inference_mode():
      # forward pass and loss
      yy,hh = net(X)
      y_hat[time+seq_length] = yy[-1]
      h[time + seq_length,:] = hh.detach()

  return y_hat,h
