function outputs = target_transform_inv(inputs)
% inputs: [H, W, 4 * N]

N = size(inputs, 3);
assert(mod(N, 4) == 0);
N = N / 4;

outputs = inputs;

for n = 1 : N
    outputs(:, :, n : N : N * 4) = inputs(:, :, (n-1) * 4 + 1 : n * 4);
end
