# based on Spotlight
# https://github.com/maciejkula/spotlight
# Copyright (c) 2017 Maciej Kula, 2019 Andrew Kane
# MIT License

require "torch"
require "csv"

data = []
CSV.foreach("ml-100k/u.data", col_sep: "\t") do |row|
  # subtract 1 from user_id and item_id so they start at zero
  data << {
    user_id: row[0].to_i - 1,
    item_id: row[1].to_i - 1,
    rating: row[2].to_f
  }
end

data.shuffle!

train_set = data.first(80000)
valid_set = data.last(20000)

# should just use train set, but keep prediction logic simple for now
n_users = data.map { |v| v[:user_id] }.max + 1
n_items = data.map { |v| v[:item_id] }.max + 1

class ScaledEmbedding < Torch::NN::Embedding
  def reset_parameters
    # sets initial weights, very important
    @weight.data.normal!(0, 1.0 / @embedding_dim)
  end
end

class MatrixFactorization < Torch::NN::Module
  def initialize(n_users, n_items, n_factors: 20)
    super()
    @user_factors = ScaledEmbedding.new(n_users, n_factors)
    @item_factors = ScaledEmbedding.new(n_items, n_factors)
  end

  def forward(user, item)
    (@user_factors.call(user) * @item_factors.call(item)).sum(1)
  end
end

model = MatrixFactorization.new(n_users, n_items, n_factors: 20)
optimizer = Torch::Optim::Adam.new(model.parameters, lr: 1e-3, weight_decay: 1e-9)
loss_func = Torch::NN::MSELoss.new

def to_tensors(data)
  [
    Torch.tensor(data.map { |v| v[:user_id] }),
    Torch.tensor(data.map { |v| v[:item_id] }),
    Torch.tensor(data.map { |v| v[:rating] })
  ]
end

valid_user, valid_item, valid_rating = to_tensors(valid_set)

10.times do |epoch|
  train_loss = 0.0
  started_at = Time.now

  batches = train_set.each_slice(64)
  batches.each do |batch|
    user, item, rating = to_tensors(batch)
    prediction = model.call(user, item)

    optimizer.zero_grad

    loss = loss_func.call(prediction, rating)
    train_loss += loss.item * batch.size

    loss.backward
    optimizer.step
  end

  train_loss /= train_set.size
  valid_loss = Torch.no_grad { loss_func.call(model.call(valid_user, valid_item), valid_rating).item }
  time = Time.now - started_at

  puts "epoch: %d, train mse: %.3f, valid mse: %.3f, time: %ds" % [epoch, train_loss, valid_loss, time]
end
