import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

''' Functions '''
def log_loss(y_pred, y):
  ''' This function computes the loss function, quantifying the similarity
  between the predicted probability and actual value '''
  cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_pred)
  return tf.reduce_mean(cross_entropy)


def predict(y_pred, threshold=0.5):
  ''' This function compares the predicted value to the threshold '''
  return tf.cast(y_pred > threshold, dtype = tf.float32) # Returns 1 if above threshold, else 0


def accuracy(y_pred, y):
  ''' This function quantifies model accuracy by comparing predicted and actual output values '''
  y_pred = predict(tf.math.sigmoid(y_pred))
  eq = tf.cast(y_pred == y, tf.float32) # Returns tensor with idx where predicted and actual values match
  acc = tf.reduce_mean(eq) # Accuracy value = average num of correct predictions
  return acc

''' Classes '''
class Normalizer(tf.Module):
  ''' This class performs normalization on an inputted dataset via z-scoring'''
  def __init__(self, data):
    # Initialize mean and standard deviation
    self.mean = tf.Variable(tf.math.reduce_mean(data))
    self.std = tf.Variable(tf.math.reduce_std(data))

  def norm(self, data):
    ''' This method normalizes the input data '''
    return (data - self.mean) / self.std

  def unnorm(self, data):
    ''' This method un-normalizes the input data '''
    return (data * self.std) + self.mean


class LogRegModel(tf.Module):
  def __init__(self):
    return

  def __call__(self, x):
    # Executes when object is called
    # x – m x n tensor
    # Randomize weight/bias values from a uniform distribution
    weight = tf.random.uniform(shape=[x.shape[-1], 1], seed=42) # n x 1
    bias = tf.random.uniform(shape=[], seed = 42) # 1 x 1
    self.W = tf.Variable(weight, name='W')
    self.b = tf.Variable(bias, name='b')
    y_pred = tf.matmul(x, self.W) + self.b # m x 1
    y_pred = tf.squeeze(y_pred, axis=-1) # m
    return y_pred


class TrainingLoop(tf.Module):
  def __init__(self):
    self.losses = {'train':[], 'test':[], 'val':[]}
    self.accs = {'train':[], 'test':[], 'val':[]}
    self.model = None
    self.num_epochs = 200
    return

  def train(self, train_data, test_data, num_epochs=None, learn_rate=0.01, model=None):
    ''' This function runs the training loop on the inputted training data '''
    self.model = LogRegModel() if model is None else model
    self.num_epochs = num_epochs if num_epochs is not None else self.num_epochs

    print("Training Started...")
    for epoch in range(self.num_epochs):
      batch_losses = {'train':[], 'test':[]} # track loss values across each batch
      batch_accs = {'train':[], 'test':[]} # track accuracy scores across each batch

      # Iterate through training data via batches
      for x_batch, y_batch in train_data:
        with tf.GradientTape() as tape:
          y_pred_batch = self.model(x_batch) # predictions from one particular batch
          batch_loss = log_loss(y_pred=y_pred_batch, y=y_batch) # loss value for this particular batch
        batch_acc = accuracy(y_pred_batch, y_batch) # accuracy score for this particular batch

        ## Update variables via gradient descent
        grads = tape.gradient(batch_loss, self.model.variables) # computed gradient
        for g, v in zip(grads, self.model.variables):
          v.assign_sub(learn_rate * g)
        
        ## Track training performance
        batch_losses['train'].append(batch_loss)
        batch_accs['train'].append(batch_acc)

      # Iterate through test data via batches
      for x_batch, y_batch in test_data:
        y_pred_batch = self.model(x_batch) # predictions from one particular batch
        batch_loss = log_loss(y_pred_batch, y_batch) # loss value for this particular batch
        batch_acc = accuracy(y_pred_batch, y_batch) # accuracy score for this particular batch

        ## Track test performance
        batch_losses['test'].append(batch_loss)
        batch_accs['test'].append(batch_acc)
      
      # Track model performance per each epoch
      self.losses['train'].append( tf.reduce_mean(batch_losses['train']) )
      self.accs['train'].append( tf.reduce_mean(batch_accs['train']) )

      self.losses['test'].append( tf.reduce_mean(batch_losses['test']) )
      self.accs['test'].append( tf.reduce_mean(batch_accs['test']) )

      # Track progress via output
      if epoch % (self.num_epochs / 10) == 0: # print 10 progress reports in total
        print(f"Epoch: {epoch}\tTraining Loss: {self.losses['train'][-1]}")
      
    print("...Complete.")
    return
  
  def plot(self):
    print("Plotting...")
    
    fig, axs = plt.subplots(1,2)
    fig.set_size_inches(w=15,h=5)

    # Plot log loss throughout each epoch
    axs[0].plot(np.arange(self.num_epochs), self.losses['train'], label='Training Loss', color = 'b')
    axs[0].plot(np.arange(self.num_epochs), self.losses['test'], label='Test Loss', color = 'r')

    axs[0].legend()
    axs[0].set_title('Log Loss vs. Epoch')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Log Loss')
    axs[0].set_xlim([0, self.num_epochs])

    # Plot accuracy throughout each epoch
    axs[1].plot(np.arange(self.num_epochs), self.accs['train'], label='Training Accuracy', color = 'b')
    axs[1].plot(np.arange(self.num_epochs), self.accs['test'], label='Test Accuracy', color = 'r')

    axs[1].legend()
    axs[1].set_title('Accuracy vs. Epoch')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].set_xlim([0, self.num_epochs])
    
    print("...Complete")
    return