function ret = cd1(rbm_w, visible_data)
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.

    % Create probabilities of hidden units
    p = visible_state_to_hidden_probabilities(rbm_w, visible_data);

    % Generate sample with this probabilities
    hs = sample_bernoulli(p);

    % Calculate the reconstruction of the visible_state
    rvp = hidden_state_to_visible_probabilities(rbm_w, hs);

    % Generate visible sample
    rvs = sample_bernoulli(rvp');

    % Generate again hidden probabilities and sample
    p = visible_state_to_hidden_probabilities(rbm_w, rvs);
    hs2 = sample_bernoulli(p);

    % Calculate the gradient
    cd1 = hs - hs2;

end
