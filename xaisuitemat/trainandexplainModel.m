function [modelfn, resultdata] = trainandexplainModel(model, datapath, target, explainers, varargin)
    data = readtable(datapath);
    Y = eval(("data." + target));
    eval(("data." + target + '= []'));
    X = data;
    train_data = X(1:floor(end-0.8*end),:);
    test_data = X(floor(0.8*end+1):end,:);
    train_target = Y(1:floor(end-0.8*end),:);
    test_target = Y(floor(0.8*end+1):end,:);
    modelfn = eval("fit" + model + "(train_data, train_target)");
    predictions = modelfn.predict(test_data);
    %directhits = sum(predictions == test_target) / length(predictions);
    performance = loss(modelfn, test_data, test_target);
    fprintf("Model %s has loss of %.2f", model, performance);
    for explainer = 1:length(explainers)
        exp = eval(explainers(explainer) + "(modelfn, X)");
        exp = fit(exp, varargin{:});
        figure;
        plot(exp);
    end
    resultdata = [test_target predictions];
end
