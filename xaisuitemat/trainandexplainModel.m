function [modelfn, resultdata, explained] = trainandexplainModel(model, data, explainers, varargin)
    explained = cell(length(explainers), 1);
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
        if isempty(varargin)
            exp = fit(exp);
        elseif length(varargin) == 1
            exp = fit(exp, data(varargin{1}, :));
        elseif length(varargin)==2 && strcmp(explainers(explainer), "lime")
            exp = fit(exp, data(varargin{1}, :), varargin{2});
        else
            exp = fit(exp, data(varargin{1}, :));
        end
        figure;
        f = plot(exp);
        b = findobj(f,'Type','bar');
        imp = b.YData;
        T = table(flipud(array2table(imp', RowNames = gcf().CurrentAxes.YTickLabel, VariableNames = {'Predictor Importance'})));
        writetable(splitvars(T), model + " " + explainers(explainer) + " " + varargin{1} + ".csv");
        explained{explainer} = struct(exp);
        set(gca,'TickLabelInterpreter','none')

    end
    resultdata = [Y_test predictions];
end
