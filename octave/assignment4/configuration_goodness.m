function G = configuration_goodness(rbm_w, visible_state, hidden_state)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a binary matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% This returns a scalar: the mean over cases of the goodness (negative energy) of the described configurations.
    
    nc = size(visible_state, 2);
    E = zeros(nc, 1);
    
    for n = 1:nc
        E(n) = hidden_state(:, n)' * rbm_w * visible_state(:, n);
    end
    
    G = mean(E);
    
end
