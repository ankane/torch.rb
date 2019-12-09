# ported from PyTorch Tutorials
# https://github.com/pytorch/tutorials/blob/master/beginner_source/nlp/sequence_models_tutorial.py
# Copyright (c) 2017 PyTorch contributors, 2019 Andrew Kane
# BSD 3-Clause License

require "torch"

Torch.manual_seed(1)

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

def prepare_sequence(seq, to_ix)
  idxs = seq.map { |w| to_ix[w] }
  return Torch.tensor(idxs, dtype: :long)
end


training_data = [
  ["The dog ate the apple".split, ["DET", "NN", "V", "DET", "NN"]],
  ["Everybody read that book".split, ["NN", "V", "DET", "NN"]]
]
word_to_ix = {}
training_data.each do |sent, tags|
  sent.each do |word|
    word_to_ix[word] ||= word_to_ix.length
  end
end
p word_to_ix
tag_to_ix = {"DET" => 0, "NN" => 1, "V" => 2}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6

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
    # Step 1. Remember that Pytorch accumulates gradients.
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
