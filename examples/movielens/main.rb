# based on Spotlight
# https://github.com/maciejkula/spotlight

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
test_set = data.last(20000)

# should just use train set, but keep prediction logic simple for now
n_users = data.map { |v| v[:user_id] }.max + 1
n_items = data.map { |v| v[:item_id] }.max + 1

class ScaledEmbedding < Torch::NN::Embedding
  def reset_parameters
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

model = MatrixFactorization.new(n_users, n_items, n_factors: 128)
optimizer = Torch::Optim::Adam.new(model.parameters, lr: 1e-3, weight_decay: 1e-9)
loss_func = Torch::NN::MSELoss.new

10.times do |epoch|
  epoch_loss = 0.0

  batches = train_set.each_slice(1024)
  batches.each do |batch|
    user = Torch.tensor(batch.map { |v| v[:user_id] })
    item = Torch.tensor(batch.map { |v| v[:item_id] })
    rating = Torch.tensor(batch.map { |v| v[:rating] })

    prediction = model.call(user, item)

    optimizer.zero_grad

    loss = loss_func.call(prediction, rating)
    epoch_loss += loss.item

    loss.backward
    optimizer.step
  end

  epoch_loss /= batches.size

  puts "Epoch %d: loss %.3f" % [epoch, epoch_loss]
end

def rmse(model, dataset)
  user = Torch.tensor(dataset.map { |v| v[:user_id] })
  item = Torch.tensor(dataset.map { |v| v[:item_id] })
  rating = Torch.tensor(dataset.map { |v| v[:rating] })

  prediction = model.call(user, item)
  Math.sqrt(((rating - prediction)**2).mean.item)
end

train_rmse = rmse(model, train_set)
test_rmse = rmse(model, test_set)

puts "Train RSME %.3f, test RMSE %.3f" % [train_rmse, test_rmse]
