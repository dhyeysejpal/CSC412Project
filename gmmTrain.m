function gmms = gmmTrain( train_data, max_iter, epsilon, M )
% gmmTain
%
%  inputs:  dir_train  : a string pointing to the high-level
%                        directory containing each speaker directory
%           max_iter   : maximum number of training iterations (integer)
%           epsilon    : minimum improvement for iteration (float)
%           M          : number of Gaussians/mixture (integer)
%
%  output:  gmms       : a 1xN cell array. The i^th element is a structure
%                        with this structure:
%                            gmm.name    : string - the name of the speaker
%                            gmm.weights : 1xM vector of GMM weights
%                            gmm.means   : DxM matrix of means (each column 
%                                          is a vector
%                            gmm.cov     : DxDxM matrix of covariances. 
%                                          (:,:,i) is for i^th mixture

% max_iter = 100, epsilon = 0.0001, M = 8


    fn_GMMs = sprintf('GMMs-i%d-e%f-m%d.mat', max_iter, epsilon, M);
    
    % Number of dimensions in a frame
    d = 1024;

    fnames = fieldnames(train_data);

    S = length(fnames);
    gmms = cell(1, S);

    for s = 1:S
        emotion = char(fnames(s));
        X = train_data.(emotion);
        
        omega = ones(1, M) ./ M;
        sigma = repmat(eye(d), 1, 1, M);
        r = randi(size(X,1), 1,M);
        for j = 1:length(r)
        	mu(:,j) = X(r(j),:)';
        end
        i = 0;
        prev_L = -Inf;
        improvement = Inf;
        while i <= max_iter && improvement >= epsilon
            [p, L] = ComputeLikelihood(X, omega, mu, sigma); % Note: the function returns log likelihood.
            [omega, mu, sigma] = UpdateParameters(X, omega, mu, sigma, p);
            improvement = L - prev_L;
            prev_L = L;
            i = i + 1;
            disp([i,improvement]);
        end
        
        gmm = struct();
        gmm.name = emotion;
        gmm.weights = omega;
        gmm.means = mu;
        gmm.cov = sigma;

        gmms{s} = gmm;  
    end
    save(fn_GMMs, 'gmms', '-mat');
end



function [p, L] = ComputeLikelihood(X, omega, mu, sigma)
    T = size(X, 1);
    M = length(omega);
    log_b = zeros(T, M);
    d = size(sigma, 1);
    for m = 1:M
        % Now we have all elements of the mth diagonal matrix as a row. Easier for computation.
        sigma_row = diag(sigma(:, :, m))';
        log_den = ((d / 2) * log(2 * pi)) + (1/2 * log(prod(sigma_row)));
        mu_mat = repmat(mu(:,m)', T, 1);    % Now it TxM matrix, with each row as a copy of the mu
        sigma_row_mat = repmat(sigma_row, T, 1);    % Will be used for division 

        % based on equation (4)
        log_b(:, m) = sum((((X - mu_mat) .^ 2) ./ sigma_row_mat), 2) .* -0.5 - log_den;
    end
    rep_omega = repmat(omega, T, 1);    % T rows, each having a copy of omega
    p = (log_b + log(rep_omega));    % Size is TxM
    max_p = max(max(p));

    % Each row has log(omega_m * b_m) for x_t.
    % We sum over columns to get probability of x_t, and p(X) = prod(p(x_t|theta))
    L = log(sum(exp(p - max_p), 2)) + max_p;
    disp(L);
    L = sum(L);   % log p(X) = sum(log(p(x_t|theta)))
    p = diag(1 ./ sum(p, 2)) * p;    % Each element has to be divided by the sum of its row
end


function [omega, mu, sigma] = UpdateParameters(X, omega, mu, sigma, p)
    T = size(p, 1);
    M = size(p, 2);
    d = size(sigma, 1);
    
    den = sum(p, 1); % 1xD dimensional. Summing over x_t for all values of t, for each p(m|x_-t, theta)
    
    omega =  den ./ T;  

    % As per equations given in the handout
    for m = 1:M
        p_m = repmat(p(:,m), 1, d);    % TxD
        mu_1 = sum((X .* p_m), 1) ./ den(1, m);    % 1xD
        mu(:,m) = mu_1';
        sigma_m = sigma(:,:,m);
        sigma_m(1:d+1:end) = (sum(((X .^ 2) .* p_m), 1) ./ den(1, m)) - (mu_1 .^ 2);
        sigma(:,:,m) = sigma_m;
    end
end
