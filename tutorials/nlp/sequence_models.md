# Sequence Models and Long Short-Term Memory Networks

Ported from [this tutorial](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html) by Robert Guthrie

Available under the same license ([BSD-3-Clause](LICENSE-nlp-tutorial.txt))

---

At this point, we have seen various feed-forward networks. That is, there is no state maintained by the network at all. This might not be the behavior we want. Sequence models are central to NLP: they are models where there is some sort of dependence through time between your inputs. The classical example of a sequence model is the Hidden Markov Model for part-of-speech tagging. Another example is the conditional
random field.

A recurrent neural network is a network that maintains some kind of state. For example, its output could be used as part of the next input, so that information can propagate along as the network passes over the sequence. In the case of an LSTM, for each element in the sequence, there is a corresponding *hidden state*, which in principle can contain information from arbitrary points earlier in the sequence. We can use the hidden state to predict words in a language model, part-of-speech tags, and a myriad of other things.

## LSTMs in Torch.rb

Before getting to the example, note a few things. Torch.rb’s LSTM expects all of its inputs to be 3D tensors. The semantics of the axes of these tensors is important. The first axis is the sequence itself, the second indexes instances in the mini-batch, and the third indexes elements of the input. We haven’t discussed mini-batching, so let’s just ignore that and assume we will always have just 1 dimension on the second axis.

Let’s see a quick example.

```ruby
# Author: Robert Guthrie

require "torch"

Torch.manual_seed(1)
```

```ruby
lstm = Torch::NN::LSTM.new(3, 3) # Input dim is 3, output dim is 3
inputs = 5.times.map { Torch.randn(1, 3) }  # make a sequence of length 5

# initialize the hidden state.
hidden = [Torch.randn(1, 1, 3),
          Torch.randn(1, 1, 3)]

inputs.each do |i|
  # Step through the sequence one element at a time.
  # after each step, hidden contains the hidden state.
  out, hidden = lstm.call(i.view(1, 1, -1), hx: hidden)
end

# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = Torch.cat(inputs).view(inputs.length, 1, -1)
hidden = [Torch.randn(1, 1, 3), Torch.randn(1, 1, 3)]  # clean out hidden state
out, hidden = lstm.call(inputs, hx: hidden)
p out
p hidden
```

Out:

```text
tensor([[[-0.0187,  0.1713, -0.2944]],

        [[-0.3521,  0.1026, -0.2971]],

        [[-0.3191,  0.0781, -0.1957]],

        [[-0.1634,  0.0941, -0.1637]],

        [[-0.3368,  0.0959, -0.0538]]], requires_grad: true)
[tensor([[[-0.3368,  0.0959, -0.0538]]], requires_grad: true), tensor([[[-0.9825,  0.4715, -0.0633]]], requires_grad: true)]
```

## Example: An LSTM for Part-of-Speech Tagging

In this section, we will use an LSTM to get part of speech tags. We will not use Viterbi or Forward-Backward or anything like that, but as a (challenging) exercise to the reader, think about how Viterbi could be used after you have seen what is going on. In this example, we also refer to embeddings. If you are unfamiliar with embeddings, you can read up about them [here](word_embeddings.md).

Prepare data:

```ruby
def prepare_sequence(seq, to_ix)
  idxs = seq.map { |w| to_ix[w] }
  return Torch.tensor(idxs, dtype: :long)
end

training_data = [
  # Tags are: DET - determiner; NN - noun; V - verb
  # For example, the word "The" is a determiner
  ["The dog ate the apple".split, ["DET", "NN", "V", "DET", "NN"]],
  ["Everybody read that book".split, ["NN", "V", "DET", "NN"]]
]
word_to_ix = {}
# For each words-list (sentence) and tags-list in each tuple of training_data
training_data.each do |sent, tags|
  sent.each do |word|
    word_to_ix[word] ||= word_to_ix.length # Assign each word with a unique index
  end
end
p word_to_ix
tag_to_ix = {"DET" => 0, "NN" => 1, "V" => 2} # Assign each tag with a unique index

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6
```

Out:

```text
{"The"=>0, "dog"=>1, "ate"=>2, "the"=>3, "apple"=>4, "Everybody"=>5, "read"=>6, "that"=>7, "book"=>8}
```

Create the model:

```ruby
class LSTMTagger < Torch::NN::Module
  def initialize(embedding_dim, hidden_dim, vocab_size, tagset_size)
    super()
    @hidden_dim = hidden_dim

    @word_embeddings = Torch::NN::Embedding.new(vocab_size, embedding_dim)

    # The LSTM takes word embeddings as inputs, and outputs hidden states
    # with dimensionality hidden_dim.
    @lstm = Torch::NN::LSTM.new(embedding_dim, hidden_dim)

    # The linear layer that maps from hidden state space to tag space
    @hidden2tag = Torch::NN::Linear.new(hidden_dim, tagset_size)
  end

  def forward(sentence)
    embeds = @word_embeddings.call(sentence)
    lstm_out, _ = @lstm.call(embeds.view(sentence.length, 1, -1))
    tag_space = @hidden2tag.call(lstm_out.view(sentence.length, -1))
    tag_scores = Torch::NN::F.log_softmax(tag_space, 1)
    tag_scores
  end
end
```

Train the model:

```ruby
model = LSTMTagger.new(EMBEDDING_DIM, HIDDEN_DIM, word_to_ix.length, tag_to_ix.length)
loss_function = Torch::NN::NLLLoss.new
optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
Torch.no_grad do
  inputs = prepare_sequence(training_data[0][0], word_to_ix)
  tag_scores = model.call(inputs)
  p tag_scores
end

300.times do |epoch|  # again, normally you would NOT do 300 epochs, it is toy data
  training_data.each do |sentence, tags|
    # Step 1. Remember that Torch.rb accumulates gradients.
    # We need to clear them out before each instance
    model.zero_grad

    # Step 2. Get our inputs ready for the network, that is, turn them into
    # Tensors of word indices.
    sentence_in = prepare_sequence(sentence, word_to_ix)
    targets = prepare_sequence(tags, tag_to_ix)

    # Step 3. Run our forward pass.
    tag_scores = model.call(sentence_in)

    # Step 4. Compute the loss, gradients, and update the parameters by
    #  calling optimizer.step()
    loss = loss_function.call(tag_scores, targets)
    loss.backward
    optimizer.step
  end
end

# See what the scores are after training
Torch.no_grad do
  inputs = prepare_sequence(training_data[0][0], word_to_ix)
  tag_scores = model.call(inputs)

  # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
  # for word i. The predicted tag is the maximum scoring tag.
  # Here, we can see the predicted sequence below is 0 1 2 0 1
  # since 0 is index of the maximum value of row 1,
  # 1 is the index of maximum value of row 2, etc.
  # Which is DET NOUN VERB DET NOUN, the correct sequence!
  p tag_scores
end
```

Out:

```text
tensor([[-1.1389, -1.2024, -0.9693],
        [-1.1065, -1.2200, -0.9834],
        [-1.1286, -1.2093, -0.9726],
        [-1.1190, -1.1960, -0.9916],
        [-1.0137, -1.2642, -1.0366]])
tensor([[-0.0462, -4.0106, -3.6096],
        [-4.8205, -0.0286, -3.9045],
        [-3.7876, -4.1355, -0.0394],
        [-0.0185, -4.7874, -4.6013],
        [-5.7881, -0.0186, -4.1778]])
```
