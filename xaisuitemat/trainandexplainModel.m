function [modelfn, resultdata] = trainandexplainModel(model, data, explainers, varargin)

    Y = data.Target;
    data.Target = [];
    X = data;
    X_train = X(1:floor(end-0.8*end),:);
    X_test = X(floor(0.8*end+1):end,:);
    Y_train = Y(1:floor(end-0.8*end),:);
    Y_test = Y(floor(0.8*end+1):end,:);
    modelfn = eval("fit" + model + "(X_train, Y_train)");
    predictions = modelfn.predict(X_test);
    %directhits = sum(predictions == test_target) / length(predictions);
    %performance = loss(modelfn, test_data, test_target);
    accuracy = corrcoef(predictions, Y_test);
    score = accuracy(1, 2);
    fprintf("%s score: %.2f", model, score);
    for explainer = 1:length(explainers)
        exp = eval(explainers(explainer) + "(modelfn, X)");
        if length(varargin) == 0
            exp = fit(exp);
        elseif length(varargin) == 1
            exp = fit(exp, data(varargin{1}, :));
        else
            exp = fit(exp, data(varargin{1}, :), varargin{2});
        end
        figure;
        plot(exp);
        set(gca,'TickLabelInterpreter','none')

    end
    resultdata = [Y_test predictions];
end
