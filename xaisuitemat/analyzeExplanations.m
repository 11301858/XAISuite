function analyzeExplanations (varargin)
    if length(varargin) < 2 
        error("There is not enough or too much data to analyze.")
    end

    for filename = 1:length(varargin)
        eval("T" + filename + "=" + "readtable(varargin{filename})")
    end


    X = corrcoef(T1.(2), T2.(2), Rows = 'complete');
    disp("Correlation in importance scores is " + X(1,2))

end
