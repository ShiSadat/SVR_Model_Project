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
function [X, y] = removeNoiseUsingZScore(X, y)
    % Remove outliers using Z-score
    [Z_X, muX, sigmaX] = zscore(X);
    [Z_y, muY, sigmaY] = zscore(y);
    
    thresholdX = 1.8; % Define threshold
    thresholdY = 0.18;
    
    disp('Z_X and Z_y with Indexes:');
    for i = 1:size(Z_X, 1)
        disp(['Index: ', num2str(i), ...
              ', Z_X: ', num2str(Z_X(i, :)), ...
              ', Z_y: ', num2str(Z_y(i))]);
    end
    % Create separate logical arrays for X and y outliers
    outliers_X = any(abs(Z_X) > thresholdX, 2);
    outliers_y = abs(Z_y) > thresholdY;
    
    % Combine X and y outliers
    outliers = outliers_X | outliers_y;
    
    % Find indices of outliers
    outlier_indices = find(outliers);
    
    % Display information about outliers
    disp('Outliers:');
    disp('Index | Outlier in X/y | Values');
    for i = 1:length(outlier_indices)
        idx = outlier_indices(i);
        outlier_type = '';
        if outliers_X(idx)
            outlier_type = 'X';
        end
        if outliers_y(idx)
            outlier_type = [outlier_type 'y'];
        end
        disp([num2str(idx), ' | Outlier in: ', outlier_type, ' | X: ', num2str(X(idx,:)), ' | Z_scoreY: ', num2str(abs(Z_y(idx))), ', y: ', num2str(y(idx))]);
    end
    
    % Remove outliers
    X = X(~outliers, :);
    y = y(~outliers);

    disp(['Removed ', num2str(sum(outliers)), ' noisy samples using Z-score method.']);
end

% Helper functions for plotting
function plotActualVsPredicted(y_true, y_pred, target, savePath)
    % Plot Actual vs Predicted
    figure;
    scatter(y_true, y_pred, 'b', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 'k--', 'LineWidth', 3);
    xlabel('Actual');
    ylabel('Predicted');
    title(['Actual vs Predicted ', strrep(target, '_', ' ')]);
    saveas(gcf, fullfile(savePath, ['actual_vs_predicted_', target, '.png']));
    close;
end

function plotResiduals(y_true, y_pred, target, savePath)
    % Plot Residuals
    residuals = y_true - y_pred;
    figure;
    scatter(y_pred, residuals, 'r', 'filled', 'MarkerFaceAlpha', 0.6);
    hold on;
    yline(0, 'k--');
    xlabel('Predicted');
    ylabel('Residuals');
    title(['Residual Plot of ', strrep(target, '_', ' ')]);
    saveas(gcf, fullfile(savePath, ['residual_plot_', target, '.png']));
    close;
end

function plotCorrelationMatrix(data, savePath)
    % Plot Correlation Matrix
    figure;
    corrMatrix = corr(data{:,:});
    heatmap(data.Properties.VariableNames, data.Properties.VariableNames, corrMatrix, 'Colormap', jet, 'ColorLimits', [-1, 1]);
    title('Correlation Matrix');
    saveas(gcf, fullfile(savePath, 'correlation_matrix.png'));
    close;
end

function predictWithSavedModel(modelFileName, newParameters)
    % Load the saved model and scaling parameters
    loadedData = load(modelFileName);
    svr = loadedData.bestModel;
    mu = loadedData.mu;
    sigma = loadedData.sigma;
    r2 = loadedData.r2;
    mse = loadedData.mse;
    % Display the loaded model details
    disp('Loaded Model Details:');
    disp(svr);
    disp('r2 :');
    disp(r2);
    disp('mse:');
    disp(mse);
    % Display the Alpha
    disp('Alpha:');
    disp(svr.Alpha);
    disp('SupportVector:');
    disp(svr.SupportVectors);
 

    % Scale the new parameters
    newParametersScaled = (newParameters - mu) ./ sigma;
    
    % Make prediction
    predictedValue = predict(svr, newParametersScaled);
    
    % Display the prediction
    disp(['Predicted Value: ', num2str(predictedValue)]);
end


% Call the function with the specified file
%SVRML('XYZ deformation-implant.csv');


function extractSVREquation(modelFileName)
    % Load the saved model and scaling parameters
    loadedData = load(modelFileName);
    svr = loadedData.bestModel;
    mu = loadedData.mu;
    sigma = loadedData.sigma;
    r2 = loadedData.r2;
    mse = loadedData.mse;

    % Display the loaded model details
    disp('Loaded Model Details:');
    disp(svr);
    disp('r2 :');
    disp(r2);
    disp('mse:');
    disp(mse);

    % Extract model parameters
    supportVectors = svr.SupportVectors;
    alpha = svr.Alpha;
    bias = svr.Bias;
    kernelScale = svr.KernelParameters.Scale;
    gamma = 1 / (2 * kernelScale^2);
    disp('alpha:');
    disp(num2str(alpha));
    disp('boxConstraints:');
    disp(num2str(svr.BoxConstraints))


    % Define symbolic variables for the features
    numFeatures = size(supportVectors, 2);
    syms x [1 numFeatures];

    % Initialize the equation with the bias term
    equation = bias;

    % Iterate over each support vector to build the equation
    for i = 1:length(alpha)
        % Compute the squared Euclidean distance
        distanceSquared = sum((supportVectors(i, :) - x).^2);
        
        % Compute the RBF kernel output
        kernelOutput = exp(-gamma * distanceSquared);
        
        % Update the equation
        equation = equation + alpha(i) * kernelOutput;
    end

    % Display the final equation
    disp('The extracted equation is:');
    disp(equation);
    
    % You can save the symbolic equation to a file if needed
    % save('extracted_equation.mat', 'equation');
end

function extractShortSVREquation(modelFileName, k)
    % Load the saved model and scaling parameters
    loadedData = load(modelFileName);
    svr = loadedData.bestModel;
    mu = loadedData.mu;
    sigma = loadedData.sigma;
    r2 = loadedData.r2;
    mse = loadedData.mse;

    % Display the loaded model details
    disp('Loaded Model Details:');
    disp(svr);
    disp('r2 :');
    disp(r2);
    disp('mse:');
    disp(mse);

    % Extract model parameters
    supportVectors = svr.SupportVectors;
    alpha = svr.Alpha;
    bias = svr.Bias;
    kernelScale = svr.KernelParameters.Scale;
    gamma = 1 / (2 * kernelScale^2);

    % Sort the alphas by their absolute values and take the top k
    [~, idx] = sort(abs(alpha), 'descend');
    topKIdx = idx(1:k);

    % Define symbolic variables for the features
    numFeatures = size(supportVectors, 2);
    syms x [1 numFeatures];

    % Initialize the equation with the bias term
    equation = bias;

    % Iterate over the top k support vectors to build the simplified equation
    for i = topKIdx'
        % Compute the squared Euclidean distance
        distanceSquared = sum((supportVectors(i, :) - x).^2);
        
        % Compute the RBF kernel output
        kernelOutput = exp(-gamma * distanceSquared);
        
        % Update the equation
        equation = equation + alpha(i) * kernelOutput;
    end

    % Display the final simplified equation
    disp(['The extracted equation with top ', num2str(k), ' support vectors is:']);
    disp(equation);
    
    % You can save the symbolic equation to a file if needed
    % save('extracted_short_equation.mat', 'equation');
end
function evaluateFullSVRModel(modelFileName, newFileName, featureColumns, targetColumn, k)
    % Load the saved model and scaling parameters
    loadedData = load(modelFileName);
    svr = loadedData.bestModel;
    mu = loadedData.mu;
    sigma = loadedData.sigma;
    original_r2 = loadedData.r2;

    disp(['Original model R² (from training): ', num2str(original_r2)]);

    % Extract model parameters
    supportVectors = svr.SupportVectors;
    alpha = svr.Alpha;
    bias = svr.Bias;
    kernelScale = svr.KernelParameters.Scale;
    gamma = 1 / (2 * kernelScale^2);

    % Sort the alphas by their absolute values and take the top k
    if k > length(alpha)
        k = length(alpha);  % Ensure k does not exceed the number of support vectors
    end
    [~, idx] = sort(abs(alpha), 'descend');
    topKIdx = idx(1:k);

    % Create symbolic variables for the equation
    syms x [1 length(featureColumns)];

    % Build the equation using the top k support vectors
    shortEquation = bias;
    for i = topKIdx'
        distanceSquared = sum((supportVectors(i, :) - x).^2);
        kernelOutput = exp(-gamma * distanceSquared);
        shortEquation = shortEquation + alpha(i) * kernelOutput;
    end

    % Print the shortened equation
    disp(['The shortened SVR equation with top ', num2str(k), ' support vectors is:']);
    disp(shortEquation);
    % Load new data from the specified Excel file
    data = readtable(newFileName, 'VariableNamingRule', 'preserve');
    
    % Add an index column to the table
    data.Index = (1:height(data))';
    
    % Move the index column to the first position for better readability
    data = movevars(data, 'Index', 'Before', 1);

    % Display the table with the index
    disp('All data with index:');
    disp(data);

    % Check if all feature columns exist in the table
    missingColumns = setdiff([featureColumns, targetColumn], data.Properties.VariableNames);
    if ~isempty(missingColumns)
        error('The following columns are missing from the data: %s', strjoin(missingColumns, ', '));
    end

    % Extract features and target
    X_new = data{:, featureColumns};
    y_new = data{:, targetColumn};
    [X_new, y_new] = removeNoiseUsingZScore(X_new, y_new);

    % Scale the new data using the same scaling parameters
    X_new_scaled = (X_new - mu) ./ sigma;

    % Predict using the full model
    y_pred_model = predict(svr, X_new_scaled);

    % Predict using the shortened equation
    y_pred_shortEquation = zeros(size(y_new));
    for i = 1:size(X_new_scaled, 1)
        y_pred_shortEquation(i) = double(subs(shortEquation, x, X_new_scaled(i, :)));
    end
    for i = 1:size(X_new_scaled,2)
                plotSVRDecisionBoundary(svr, X_new(:,i),y_new , targetColumn,'S:\SHIMA\Project\SVR MODEL' )
    end
    
    % Compute the accuracy of the full model and shortened equation on the new data
    mse_model = mean((y_new - y_pred_model).^2);
    r2_model = 1 - sum((y_new - y_pred_model).^2) / sum((y_new - mean(y_new)).^2);
    
    mse_shortEquation = mean((y_new - y_pred_shortEquation).^2);
    r2_shortEquation = 1 - sum((y_new - y_pred_shortEquation).^2) / sum((y_new - mean(y_new)).^2);

    % Display the accuracy metrics
    disp('Performance on new data using the full model:');
    disp(['Mean Squared Error: ', num2str(mse_model)]);
    disp(['R-squared Score: ', num2str(r2_model)]);
    
    disp(['Performance on new data using the shortened equation (top ', num2str(k), ' support vectors):']);
    disp(['Mean Squared Error: ', num2str(mse_shortEquation)]);
    disp(['R-squared Score: ', num2str(r2_shortEquation)]);

    % Optional: Plot predicted vs actual values
    figure;
    scatter(y_new, y_pred_model, 'b', 'DisplayName', 'Model Prediction');
    hold on;
    scatter(y_new, y_pred_shortEquation, 'r', 'DisplayName', 'Shortened Equation Prediction');
    plot([min(y_new), max(y_new)], [min(y_new), max(y_new)], 'k--', 'DisplayName', 'Perfect Prediction');
    xlabel('Actual Values');
    ylabel('Predicted Values');
    title('Predicted vs Actual Values');
    legend('Location', 'best');
    hold off;
end

function illustrateFeatureDependence(modelFileName, newFileName, featureColumns, targetColumn, k)
    % Load the saved model and scaling parameters
    loadedData = load(modelFileName);
    svr = loadedData.bestModel;
    mu = loadedData.mu;
    sigma = loadedData.sigma;
    original_r2 = loadedData.r2;

    disp(['Original model R² (from training): ', num2str(original_r2)]);

    % Extract model parameters
    supportVectors = svr.SupportVectors;
    alpha = svr.Alpha;
    bias = svr.Bias;
    kernelScale = svr.KernelParameters.Scale;
    gamma = 1 / (2 * kernelScale^2);

    % Sort the alphas by their absolute values and take the top k
    if k > length(alpha)
        k = length(alpha);  % Ensure k does not exceed the number of support vectors
    end
    [~, idx] = sort(abs(alpha), 'descend');
    topKIdx = idx(1:k);

    % Create symbolic variables for the equation
    syms x [1 length(featureColumns)];

    % Build the equation using the top k support vectors
    shortEquation = bias;
    for i = topKIdx'
        distanceSquared = sum((supportVectors(i, :) - x).^2);
        kernelOutput = exp(-gamma * distanceSquared);
        shortEquation = shortEquation + alpha(i) * kernelOutput;
    end

    % Print the shortened equation
    disp(['The shortened SVR equation with top ', num2str(k), ' support vectors is:']);
    disp(shortEquation);

    % Load new data from the specified Excel file
    data = readtable(newFileName, 'VariableNamingRule', 'preserve');

    % Add an index column to the table
    data.Index = (1:height(data))';

    % Move the index column to the first position for better readability
    data = movevars(data, 'Index', 'Before', 1);

    % Display the table with the index
    disp('All data with index:');
    disp(data);

    % Check if all feature columns exist in the table
    missingColumns = setdiff([featureColumns, targetColumn], data.Properties.VariableNames);
    if ~isempty(missingColumns)
        error('The following columns are missing from the data: %s', strjoin(missingColumns, ', '));
    end

    % Extract features and target
    X_new = data{:, featureColumns};
    y_new = data{:, targetColumn};
    [X_new, y_new] = removeNoiseUsingZScore(X_new, y_new);

    % Scale the new data using the same scaling parameters
    X_new_scaled = (X_new - mu) ./ sigma;

    % Predict using the full model
    y_pred_model = predict(svr, X_new_scaled);

    % Predict using the shortened equation
    y_pred_shortEquation = zeros(size(y_new));
    for i = 1:size(X_new_scaled, 1)
        y_pred_shortEquation(i) = double(subs(shortEquation, x, X_new_scaled(i, :)));
    end

    % Compute the accuracy of the full model and shortened equation on the new data
    mse_model = mean((y_new - y_pred_model).^2);
    r2_model = 1 - sum((y_new - y_pred_model).^2) / sum((y_new - mean(y_new)).^2);

    mse_shortEquation = mean((y_new - y_pred_shortEquation).^2);
    r2_shortEquation = 1 - sum((y_new - y_pred_shortEquation).^2) / sum((y_new - mean(y_new)).^2);

    % Display the accuracy metrics
    disp('Performance on new data using the full model:');
    disp(['Mean Squared Error: ', num2str(mse_model)]);
    disp(['R-squared Score: ', num2str(r2_model)]);

    disp(['Performance on new data using the shortened equation (top ', num2str(k), ' support vectors):']);
    disp(['Mean Squared Error: ', num2str(mse_shortEquation)]);
    disp(['R-squared Score: ', num2str(r2_shortEquation)]);

    % Optional: Plot predicted vs actual values
    figure;
    scatter(y_new, y_pred_model, 'b', 'DisplayName', 'Model Prediction');
    hold on;
    scatter(y_new, y_pred_shortEquation, 'r', 'DisplayName', 'Shortened Equation Prediction');
    plot([min(y_new), max(y_new)], [min(y_new), max(y_new)], 'k--', 'DisplayName', 'Perfect Prediction');
    xlabel('Actual Values');
    ylabel('Predicted Values');
    title('Predicted vs Actual Values');
    legend('Location', 'best');
    hold off;

    % Illustrate the effect of each feature independently
    for j = 1:length(featureColumns)
        featureName = featureColumns{j};
        disp(['Illustrating effect of ', featureName]);

        figure;
        hold on;
        xlabel('Feature Value');
        ylabel('Predicted Value');
        title(['Effect of ', strrep(featureName, '_', ' '), ' on ', targetColumn]);

        % Vary the selected feature while keeping others at their mean
        X_varied = mean(X_new_scaled);
        X_varied = repmat(X_varied, 100, 1);

        % Generate values for the current feature
        featureRange = linspace(min(X_new_scaled(:, j)), max(X_new_scaled(:, j)), 100);
        X_varied(:, j) = featureRange;

        % Predict using the full model
        y_varied = predict(svr, X_varied);

        % Plot the effect
        plot(featureRange, y_varied, '-b', 'LineWidth', 2);

        saveas(gcf, fullfile(pwd, ['Effect_of_', featureName, '_on_', targetColumn, '.png']));
        hold off;
    end
end
% Helper function to plot SVR decision boundary and margins
function plotSVRDecisionBoundary(model, X, y, target, savePath)
    % Ensure X is a column vector (single feature) and y is the target variable
    if size(X, 2) ~= 1
        error('The feature matrix X must have exactly one column.');
    end
    % Create a range of feature values for plotting the regression line
    X_range = linspace(min(X), max(X), 100)';

    % Plotting
    figure;
    hold on;

    % Scatter plot of data points
    scatter(X, y, 'b', 'filled', 'MarkerFaceColor', 'b');

    % Plot the SVR regression line
    plot(X, y, 'r-', 'LineWidth', 2);

    % Plot the epsilon margins
    epsilon = model.Epsilon;
    plot(X, y + epsilon, 'k--', 'LineWidth', 1);
    plot(X, y - epsilon, 'k--', 'LineWidth', 1);

    % Plot the support vectors
    scatter(supportVectors, supportVectorTargets, 'ko', 'MarkerSize', 8, 'LineWidth', 1.5);

    % Labels and Title
    xlabel('Feature');
    ylabel('Target');
    title(['SVR Decision Boundary with Support Vectors for ', strrep(target, '_', ' ')]);
    legend('Data Points', 'SVR Prediction', 'Epsilon Margin', 'Support Vectors', 'Location', 'Best');
    
    % Save the plot
    saveas(gcf, fullfile(savePath, ['svr_decision_boundary_', target, '.png']));
    close;
end
