function SVRML(fileName)
    % Load the data
    data = readtable(fileName, 'VariableNamingRule', 'preserve');
    disp('Column names in the data:');
    disp(data.Properties.VariableNames);

    % Define feature columns
    featureColumns = {'Thread pitch', 'Thread depth', 'Alpha1', 'Alpha2', 'Thread width'};

    % Check if all feature columns exist in the table
    missingColumns = setdiff(featureColumns, data.Properties.VariableNames);
    if ~isempty(missingColumns)
        error('The following feature columns are missing from the data: %s', strjoin(missingColumns, ', '));
    end
    
    X = data{:, featureColumns}; 
    targetColumns = setdiff(data.Properties.VariableNames, featureColumns);

    % Create output directory
    [~, name, ~] = fileparts(fileName);
    baseSavePath = fullfile(pwd, ['output_', name]);
    if ~exist(baseSavePath, 'dir')
        mkdir(baseSavePath);
    end

    % Create directory for best models
    bestBaseSavePath = fullfile(pwd, ['output_', name, '_best']);
    if ~exist(bestBaseSavePath, 'dir')
        mkdir(bestBaseSavePath);
    end

    % Iterate over each target column
    for i = 1:length(targetColumns)
        target = targetColumns{i};
        y = data{:, target};
        
        % Outlier detection using Z-Score
        [X_filtered, y_filtered] = removeNoiseUsingZScore(X, y);

        % Check if X and y are empty after outlier removal
        if isempty(X_filtered) || isempty(y_filtered)
            warning('All data removed after outlier detection for target %s. Skipping this target.', target);
            continue;
        end

        % Split the data
        cv = cvpartition(length(y_filtered), 'HoldOut', 0.1);
        X_train = X_filtered(training(cv), :);
        X_test = X_filtered(test(cv), :);
        y_train = y_filtered(training(cv));
        y_test = y_filtered(test(cv));

        % Scale the features
        [X_train_scaled, mu, sigma] = zscore(X_train);
        X_test_scaled = (X_test - mu) ./ sigma;

        % Hyperparameter tuning for SVR using Grid Search
        bestModel = tuneSVRHyperparameters(X_train_scaled, y_train);

        % Display best model details
        disp(['Best Parameters for ', target, ': C=', num2str(bestModel.ModelParameters.BoxConstraint), ...
              ', Epsilon=', num2str(bestModel.ModelParameters.Epsilon)]);
        disp('Best Model Details:');
        disp(bestModel);

        % Extract and display model parameters
        kernelParameters = bestModel.KernelParameters;
        disp(['Kernel Parameters:']);
        disp(kernelParameters);
        
        % Display support vectors and their coefficients
        disp('Support Vectors:');
        disp(bestModel.Alpha);

        % Evaluate the best model
        y_pred = predict(bestModel, X_test_scaled);
        mse = mean((y_test - y_pred).^2);
        r2 = 1 - sum((y_test - y_pred).^2) / sum((y_test - mean(y_test)).^2);

        disp(['Mean Squared Error for ', target, ': ', num2str(mse)]);
        disp(['R-squared Score for ', target, ': ', num2str(r2)]);

        % Predict stress for a new thread design
        new_design = [1.0, 0.3, 70, 75, 0.2];
        new_design_scaled = (new_design - mu) ./ sigma;
        predicted_stress = predict(bestModel, new_design_scaled);
        disp(['Predicted Stress for New Design (', target, '): ', num2str(predicted_stress)]);

        % Save the best model if R-squared is above threshold
        if r2 > 0.8
            modelFileName = fullfile(bestBaseSavePath, [name, '_best_svr_model_', target, '.mat']);
            % Correlation heatmap for the whole dataset
            plotCorrelationMatrix(data, bestBaseSavePath);
            % Residual plot
            plotResiduals(y_test, y_pred, target, bestBaseSavePath);
            % Visualization of actual vs predicted values
            plotActualVsPredicted(y_test, y_pred, target, bestBaseSavePath);
        else 
            modelFileName = fullfile(baseSavePath, [name, '_svr_model_', target, '.mat']);
            plotCorrelationMatrix(data, baseSavePath);
            % Residual plot
            plotResiduals(y_test, y_pred, target, baseSavePath);
            % Visualization of actual vs predicted values
            plotActualVsPredicted(y_test, y_pred, target, baseSavePath);
        end

        save(modelFileName, 'bestModel', 'mu', 'sigma','r2','mse');
        disp(['Model saved to ', modelFileName]);
    end
end

function bestModel = tuneSVRHyperparameters(X, y)
    % Tune SVR hyperparameters using Grid Search
    cv_tune = cvpartition(length(y), 'KFold', min(10, floor(size(X, 1) / 2)));
    C_values = logspace(-3, 3, 10);  % More granular range for C
    epsilon_values = logspace(-4, 0, 10);  % More granular range for epsilon
    bestScore = -Inf;
    bestModel = [];

    for C = C_values
        for epsilon = epsilon_values
            avg_r2 = crossValSVR(X, y, cv_tune, C, epsilon);
            if avg_r2 > bestScore
                bestScore = avg_r2;
                bestModel = fitrsvm(X, y, 'KernelFunction', 'rbf', ...
                                  'BoxConstraint', C, 'Epsilon', epsilon, 'KernelScale', 'auto');
            end
        end
    end
end
function avg_r2 = crossValSVR(X, y, cv_tune, C, epsilon)
    % Perform cross-validation for SVR and calculate average R-squared
    r2_scores = zeros(cv_tune.NumTestSets, 1);

    for fold = 1:cv_tune.NumTestSets
        X_train_fold = X(cv_tune.training(fold), :);
        X_test_fold = X(cv_tune.test(fold), :);
        y_train_fold = y(cv_tune.training(fold));
        y_test_fold = y(cv_tune.test(fold));

        % Train SVR model
        svr = fitrsvm(X_train_fold, y_train_fold, 'KernelFunction', 'rbf', ...
                      'BoxConstraint', C, 'Epsilon', epsilon, 'KernelScale', 'auto');

        % Make predictions
        y_pred_fold = predict(svr, X_test_fold);
        r2_scores(fold) = 1 - sum((y_test_fold - y_pred_fold).^2) / sum((y_test_fold - mean(y_test_fold)).^2);
    end

    avg_r2 = mean(r2_scores);
end
