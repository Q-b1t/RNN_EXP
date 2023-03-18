import matplotlib.pyplot as plt
import numpy as np
from IPython.display import set_matplotlib_formats
set_matplotlib_formats("svg")

def plot_sequence(t,x,title = "Target Sequence",x_label = "t",y_label = "x(t)"):
  """
  Helper function to visualize a target data sequence given the timesteps and the computer values for the sequence's function.
  """
  plt.figure(figsize = (15,4))
  plt.plot(t,x,"ks-",markerfacecolor = "w")
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()

def plot_slice_of_sequence(x,seq_length,initial_timestep = 0,x_label = "t",y_label = "x(t)"):
  """
  Helper function to check to test different sampling sequence length values (is this sequence of data enough to extrapolate the rest????)
  """
  plt.figure()
  plt.plot(x[initial_timestep:seq_length],"ks-",markerfacecolor="w")
  title = f"First {seq_length} data values" if initial_timestep == 0 else f"Sequence between {initial_timestep} and {initial_timestep+seq_length} timesteps."
  plt.title(title)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  plt.show()


def plot_loss_decay(losses):
  """
  Helper function to visualize the loss decay accross all the epochs. 
  """
  plt.figure()
  plt.plot(losses,"s-")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.title("Model Loss")
  plt.show()


def compare_sequences(data,y_hat,seq_length):
  """
  Helper function that generates visual graphs useful for comparing the input sequence and the target sequence.
  """
  plt.figure(figsize=(16,4))
  plt.subplot(1,3,1)
  plt.plot(data,"b",label="Actual Data")
  plt.plot(y_hat,"r",label = "Predicted Data")
  plt.ylim([-1.1,1.1])
  plt.legend()
  plt.subplot(1,3,2)
  plt.plot(data-y_hat,"k^")
  plt.ylim([-1.1,1.1])
  plt.title("Sign Accuracy")
  plt.subplot(1,3,3)
  plt.plot(data[seq_length:],y_hat[seq_length:],'mo')
  plt.xlabel("Real Data")
  plt.ylabel("Predicted Data")
  r = np.corrcoef(data[seq_length:],y_hat[seq_length:])
  plt.title(f"r = {r[0,1]:.2f}")
  plt.show()


def plot_rnn_weights(net,figure):
  plt.figure()
  plt.bar(range(num_hidden),net.rnn.weight_ih_l0.detach())
  plt.ylabel("Weight Value")
  plt.show()
