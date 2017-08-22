function d_G_by_rbm_w = configuration_goodness_gradient(visible_state, hidden_state)
% <visible_state> is a binary matrix of size <number of visible units> by <number of configurations that we're handling in parallel>.
% <hidden_state> is a (possibly but not necessarily binary) matrix of size <number of hidden units> by <number of configurations that we're handling in parallel>.
% You don't need the model parameters for this computation.
% This returns the gradient of the mean configuration goodness (negative energy, as computed by function <configuration_goodness>) with respect to the model parameters. Thus, the returned value is of the same shape as the model parameters, which by the way are not provided to this function. Notice that we're talking about the mean over data cases (as opposed to the sum over data cases).
    
    nc = size(visible_state, 2);
    nv = size(visible_state, 1);
    nh = size(hidden_state, 1);
    E = zeros(nh, nv);
    En = zeros(nh, nv);
    
    for n = 1:nc
      for h = 1:nh
        for v = 1:nv
          En(h, v) = hidden_state(h, n) * visible_state(v, n);
        end
      end
      
      E = E + En;
      
    end
    
    d_G_by_rbm_w = E / nc;
    
end
