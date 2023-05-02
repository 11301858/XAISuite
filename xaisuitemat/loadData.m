function data = loadData (filepath, target, varargin)
    data = readtable(filepath);
    data = renamevars(data,target,"Target");
    if length(varargin)~=0
        eval("data." + varargin{1} + "= []");
    end
end
