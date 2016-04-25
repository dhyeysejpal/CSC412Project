function gmmClassify(test_data, gmms)
	

	for i = 1:size(test_data,1)
		X = importdata(test_dir(i).name);      % Getting the content of this test file
		
		% Geting the likelihoods from all speaker models
        results = cell(10, 2);
        
		for j = 1:10
			gmm = gmms{};
			L = ComputeLikelihood(X, gmm.weights, gmm.means, gmm.cov);
			results(j,:) = [{gmm.name}, {L}];
		end
		results = sortrows(results, -2);    % Sorting in descending order
		
        disp([test_dir(i).name results{1,1}]);
		% Writing the top 5 results to a file
% 		f = fopen(strrep(test_dir(i).name, '.mfcc', '.lik'), 'w');
% 		for k = 1:5
% 			fprintf(f, '%s\t%.3f\n',results{k,1}, results{k,2});
% 		end
% 		fclose(f);
	end
end


function L = ComputeLikelihood(X, omega, mu, sigma)
    T = size(X, 1);
    M = length(omega);
    log_b = zeros(T, M);
    d = size(sigma, 1);
    for m = 1:M
        % Now we have all elements of the mth diagonal matrix as a row. Easier for computation
        sigma_row = diag(sigma(:, :, m))';
        den = (2 * pi) ^ (d / 2) * (prod(sigma_row)) ^ 0.5;    % denominator for each x_t
        log_den = log(den);

        mu_mat = repmat(mu(:,m)', T, 1);    % Now it TxM matrix, with each row as a copy of the mu
        sigma_row_mat = repmat(sigma_row, T, 1);    % Will be used for division 

        % based on the equation (4)
        log_b(:, m) = sum((((X - mu_mat) .^ 2) ./ sigma_row_mat), 2) .* -0.5 - log_den;
    end

    rep_omega = repmat(omega, T, 1);    % T rows, each having a copy of omega
    p = (exp(log_b) .* rep_omega);    % Size is TxM

    % Each row has omega_m * b_m for x_t. We sum over columns to get probability of x_t.
    L = sum(p, 2);
    L = sum(log(L));   % log p(X) = sum(log(p(x_t|theta)))
end
