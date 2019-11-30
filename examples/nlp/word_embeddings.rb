# ported from PyTorch Tutorials
# https://github.com/pytorch/tutorials/blob/master/beginner_source/nlp/word_embeddings_tutorial.py
# Copyright (c) 2017 PyTorch contributors, 2019 Andrew Kane
# BSD 3-Clause License

require "torch"

Torch.manual_seed(1)

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10
# We will use Shakespeare Sonnet 2
test_sentence = "When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.".split

# we should tokenize the input, but we will ignore that for now
# build a list of tuples.  Each tuple is ([ word_i-2, word_i-1 ], target word)
trigrams = (test_sentence.length - 2).times.map { |i| test_sentence[i, 3] }
# print the first 3, just so you can see what they look like
p trigrams.first(3)

vocab = test_sentence.to_set
word_to_ix = vocab.map.with_index.to_h

class NGramLanguageModeler < Torch::NN::Module
  def initialize(vocab_size, embedding_dim, context_size)
    super()
    @embeddings = Torch::NN::Embedding.new(vocab_size, embedding_dim)
    @linear1 = Torch::NN::Linear.new(context_size * embedding_dim, 128)
    @linear2 = Torch::NN::Linear.new(128, vocab_size)
  end

  def forward(inputs)
    embeds = @embeddings.call(inputs).view(1, -1)
    out = Torch::NN::F.relu(@linear1.call(embeds))
    out = @linear2.call(out)
    Torch::NN::F.log_softmax(out, 1)
  end
end

losses = []
loss_function = Torch::NN::NLLLoss.new
model = NGramLanguageModeler.new(vocab.size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = Torch::Optim::SGD.new(model.parameters, lr: 0.001)

10.times do |epoch|
  total_loss = 0
  trigrams.each do |trigram|
    context = trigram.first(2)
    target = trigram.last

    # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
    # into integer indices and wrap them in tensors)
    context_idxs = Torch.tensor(context.map { |w| word_to_ix[w] }, dtype: :long)

    # Step 2. Recall that torch *accumulates* gradients. Before passing in a
    # new instance, you need to zero out the gradients from the old
    # instance
    model.zero_grad

    # Step 3. Run the forward pass, getting log probabilities over next
    # words
    log_probs = model.call(context_idxs)

    # Step 4. Compute your loss function. (Again, Torch wants the target
    # word wrapped in a tensor)
    loss = loss_function.call(log_probs, Torch.tensor([word_to_ix[target]], dtype: :long))

    # Step 5. Do the backward pass and update the gradient
    loss.backward
    optimizer.step

    # Get the Ruby number from a 1-element Tensor by calling tensor.item
    total_loss += loss.item
  end

  losses << total_loss
end
p losses  # The loss decreased every iteration over the training data!
