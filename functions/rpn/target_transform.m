function outputs = target_transform(inputs)
% inputs: [H, W, 4 * N]
% outputs: [H, W, 4 * N]

N = size(inputs, 3);
assert(mod(N, 4) == 0);
N = N / 4;

outputs = inputs;

for n = 1 : N
    outputs(:, :, (n-1) * 4 + 1 : n * 4) = inputs(:, :, n : N : N * 4);
end
