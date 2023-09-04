from .xaisuiteFoundation import*

#Class XAIData starts here
class XAIData:
  '''
  Class to represent data. 

  :param numpy.ndarray X_train: The training features
  :param numpy.ndarray X_test: The testing features
  :param numpy.ndarray y_train: The training targets
  :param numpy.ndarray y_test: The testing targets
  '''
  def __init__(self, X_train, X_test, y_train, y_test):
    self.X_train = X_train
    self.X_test = X_test
    self.y_train = y_train
    self.y_test = y_test
    self.trainingData = (X_train, y_train)
    self.testingData = (X_test, y_test)

#Class Data ends here

#Class DataLoader starts here

class DataLoader:
  '''
  Class that loads data from a given source

  :param Union[str, Callable, numpy.ndarray, pd.DataFrame, tuple] data: The data identifier, a function that returns the data, or the data itself in the form of a numpy array, pandas DataFrame, or tuple
  :param str, optional source: The source of the data. Either "auto", "system", "preloaded", "generated", or "url". If "auto", the source will be inferred based on `data`. By default, "auto"
  :param str, optional type: The type of data. Either "Tabular", "Image", or "Text". By default, "Tabular". If "Text" or "Image", only one feature is allowed. 
  :param Union[str, list], optional variable_names: The variables in the dataset excluding `cut`, in the order that they appear in the data. By default, set to "auto" and inferred. 
  :param Union[str, int], optional target_names: The target variable(s). By default, set to "auto" and inferred
  :param Union[str, list], optional cut: Variables to drop from the data. By default, None. 
  :param Union[str, list], optional categorical: If type == "Tabular", variables that contain categorical data.
  :param `**dataGenerationArgs`: Additional arguments to pass in if `data` is Callable. 
  :raises ValueError: if data cannot be resolved or if invalid arguments are passed. 
  '''
  def __init__(self, data:Union[str, Callable, numpy.ndarray, pd.DataFrame, tuple], source:str = "auto", type:str = "Tabular", variable_names:Union[str, list] = "auto", target_names:Union[str, int] = "auto", cut:Union[str, list] = None, categorical:Union[str, list] = None, **dataGenerationArgs):
    '''
    Class constructor
    '''
    #Initially, we set the data content to None. self.content is meant to be a pd.DataFrame
    self.type = type
    self.content = None
    #If data is a function that returns the dataset or a string representation of such a function, we reassign data to the return value of this function. 
    if isinstance(data, Callable):
        data = data(**dataGenerationArgs) if dataGenerationArgs is not None else data()


    #If data is still a function, that means that the original value provided for data did not return a valid data value. 
    if isinstance(data, Callable):
        raise ValueError("Callable passed to DataLoader returns another Callable instead of a string, numpy.ndarray, pd.DataFrame, or tuple.")

    #If data is a DataFrame, the work is already done for us. 
    if isinstance(data, pd.DataFrame):
        self.content = data
    #Simple case of passing the provided numpy array directly to the DataFrame constructor
    elif isinstance(data, numpy.ndarray):
        self.content = pd.DataFrame(data)
    #We assume the tuple is of the form (X, y), where X and y are individually a valid data type, such as numpy.ndarray or pandas.DataFrame
    elif isinstance(data, tuple):
        self.content = pd.DataFrame(data[0])
        self.content["target"] = data[1]
    #If the data is a string, there are multiple possibilities that we need to check. 
    elif isinstance(data, str):
        match source:
            case "system":
                self.initializeDataFromSystem(data) #The source is system, so we search for a file
            case "preloaded":
                self.initializeDataFromPreloaded(data) #The source is preloaded, so we search for the data in a preloaded dictionary
            case "generated":
                self.initializeDataFromGenerated(data, **dataGenerationArgs) if dataGenerationArgs is not None else self.initializeDataFromGenerated(data, None) #The source is generated, and the string is the name of a function, so we pass data and the variables to pass to the data-generating function
            case "url":
                self.initializeDataFromUrl(data) #The source is a url, so we try to get the data from the url
            case "auto": #Here, the user is either unaware of where to look or just lazy :) Anyway, we need to do some hard work. We need to try each data loading method and see what sticks.
                try:
                    self.initializeDataFromSystem(data) 
                except:
                    try:
                      self.initializeDataFromPreloaded(data)
                    except:
                        try:
                            self.initializeDataFromGenerated(data, **dataGenerationArgs) if dataGenerationArgs is not None else self.initializeDataFromGenerated(data, None)
                        except:
                            try:
                                  self.initializeDataFromUrl(data)
                            except:
                                  raise ValueError("Data is not valid. Please make sure your data string is not misspelt and exists.") #This means we could not find the data anywhere.
                  
    assert isinstance(self.content, pd.DataFrame), "A problem occurred with the data loading. If the problem persists, file an issue at github.com/11301858/XAISuite" #By this time, data should definitely be a dataframe. If it is not, something has gone horribly wrong. 

    if cut is not None:
        self.content.drop(cut, axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove the cut variable (if it exists)

    #Now, we should have a dataframe with the features and the target. If the data type is image or text, there should be only one feature

    if (type == "Image" or type == "Text") and len(variable_names) < 2:
        return ValueError("For Image and Text data, there must be only one feature.")

    #Make sure that variable_names are the same length as the number of columns in the data provided. 

    if variable_names != "auto" and (len(variable_names) != len(self.content.columns)):
        raise ValueError("The length of variable_names is incompatible with the data provided.")

    #Make sure that the target variable, if provided, is in the provided variable_names, if provided

    if target_names != "auto" and variable_names != "auto" and target_names not in variable_names:
        raise ValueError("Custom target variable name not found in variable_names.")

    #If the target variable is auto, we determine what the value of the target variable will be first:
    if target_names == "auto" and "target" in self.content.columns:
        target_names = "target"
    elif target_names == "auto" and "target" not in self.content.columns:
        target_names = self.content.columns[-1]

    #If variable_names is auto and target is not auto, there is nothing that needs to be done

    

    #If variable_names is not auto and target is auto, we need to make sure that variable_names does not override the auto target value. 
    if variable_names != "auto" and target_names == "auto":
        target_names = variable_names[-1] if not isinstance(variable_names, str) else variable_names
    elif variable_names != "auto" and target_names == "target":
        target_names = variable_names[self.content.columns.get_loc("target")]

    #Now we set the variable_names.
    if variable_names != "auto": #Do nothing if the variable_names are set to auto
        self.content.columns = variable_names

    if categorical is not None and categorical not in variable_names:
        raise ValueError("Categorical variables provided are incompatible with variable_names.")

    #Now we split the data to make it easier to handle:

    self.y = pd.DataFrame(self.content[target_names])
    self.X = pd.DataFrame(self.content.drop(target_names, axis=1))
    self.target = target_names

    #Now we're ready to finalize creating the data object
    self.wrappedData = None
    if type == "Tabular":
        self.wrappedData = Tabular(data = self.content, categorical_columns = categorical, target_column = self.target)
    elif type == "Image":
        self.wrappedData = Image(data = self.X.to_numpy(), batched = True)
    elif type == "Text":
        self.wrappedData = Text(data = self.X.values.reshape(-1,).tolist())
    return

  def initializeDataFromSystem(self, id:str):
    '''
    Initializes data from user's system
  
    :param str id: The data file path
    :raises ValueError: if provided file path is not found or is not a file
    '''
    if not os.path.isfile(id):
        raise ValueError("Given data id is not a file path.")
    
    df = pd.read_csv(id)
    self.content = df
    return
    
  
  def initializeDataFromPreloaded(self, id:str):
    '''
    Initializes data from preloaded sklearn datasets
  
    :param str id: The name of the preloaded dataset
    :raises ValueError: if provided preloaded data name is not found
    '''
    if id in acceptedDataIDs:
        data = eval(acceptedDataIDs.get(id) + "(return_X_y=True)")
        self.content = pd.DataFrame(data[0])
        self.content["target"] = data[1]
    else:
        raise ValueError("Not an accepted preloaded data id. Available preloaded data ids are: \n" + acceptedDataIDs.keys() + "\n")
  
    return 

  def initializeDataFromGenerated(self, id:str, **generateArgs):
    '''
    Initializes data from string commands
  
    :param str id: The string generation command
    :param `**generateArgs`: Arguments to pass to the function represented by `id`
    '''
    data = eval(id + "(**generateArgs)") if generateArgs is not None else eval(id + "()")
    if isinstance(data, tuple): #We take care of the case that the generator function returns a tuple
        temp = pd.DataFrame(data[0])
        temp["target"] = data[1]
        self.content = temp
    else:
        self.content = pd.DataFrame(data)

    return  
        
  
  def initializeDataFromUrl(self, id:str):
    '''
    Initializes data from URL
  
    :param str id: The url of the data
    '''
    df = pd.read_csv(id)
    self.content = df
    return

  def plot(self):
    '''
    Plots loaded data.
    '''
    fig, axes = plt.subplots(ncols=4, nrows=int(len(self.content.columns)/4) + 1, figsize=(20, 10))
    
    for i, ax in zip(range(len(self.content.columns)), axes.flat):
        sns.histplot(self.content[self.content.columns[i]], ax=ax)
    num_extraPlots = 4 - len(self.content.columns)%4
    
    for i in range (1, num_extraPlots + 1):
      fig.delaxes(axes[int(len(self.content.columns)/4), len(self.content.columns)%4 + i - 1])
    plt.show()
    
#Class DataLoader ends here

#Class DataProcessor starts

class DataProcessor:
  '''
  Class that processes data
  
  :param DataLoader forDataLoader: The dataloader that will be associated with this processor. 
  :param float, optional test_size: The proportion of data that will be used to test and score the machine learning model. By default, 0.2
  :param Any, optional processor: The data processer, either a string function, or an Object with fit() and transform() methods.
  :param `**processorArgs`: Arguments to be passed into the processor. If the argument is a function, like a component of a composite processor, pass it in as shown in this example: DataProcessor(..., target_transform = "component: KBins(n_bins = 5)", ratio = 0.1)
  '''

  def __init__(self, forDataLoader:DataLoader, test_size:float = 0.2, processor:Any = None, **processorArgs):
    self.processor = processor
    self.processedX = None
    self.processedy = None
    self.loader = forDataLoader
    target_transform = None

    if processor is None: #Get default processor if processor is not provided
        match forDataLoader.type:
            case "Tabular":
                  processor = "TabularTransform"
            case "Image":
                  processor = "Scale"
            case "Text":
                  processor = "Tfidf"
    
    tempProcessorArgs = {}

    if processorArgs is not None: #This is if the user passes in a component transformer as a string representation of a function
        for variable in processorArgs.items():
            if isinstance(variable[1], str) and "component:" in variable[1].replace(" ", ""):
                function_string = (variable[1].replace(" ", "")).split(":", 1)[1]
                if not function_string.endswith(")"):
                    function_string = function_string + "()"
                actualValue = eval(function_string)
                tempProcessorArgs.update((variable[0], actualValue))
            else:
                tempProcessorArgs.update(variable)
    
    processorArgs = tempProcessorArgs
    
    if forDataLoader.type != "Tabular" and processorArgs.get("target_transform") is not None:
        target_transform = processorArgs.get("target_transform")
        processorArgs.pop("target_transform")
        target_transform.fit(forDataLoader.y)
        self.processedy = target_transform.transform(forDataLoader.y)
    else if forDataLoader.type != "Tabular" and processorArgs.get("target_transform") is None:
        self.processedy = target_transform.transform(forDataLoader.y)
      
        
    if isinstance(processor, str):
        self.processor = eval(processor + "(**processorArgs)") if processorArgs is not None else eval(processor + "()")
    elif isinstance(processor, Callable):
        self.processor = processor(**processorArgs) if processorArgs is not None else processor()
    else:
      self.processor = processor

    self.processor.fit(forDataLoader.wrappedData)
    processedData = self.processor.transform(forDataLoader.wrappedData)
    if forDataLoader.type != "Tabular":
            self.processedX = processedData
    else:
        tempProcessedData = pd.DataFrame(processedData, columns = forDataLoader.content.columns)
        self.processedy = tempProcessedData[forDataLoader.target].to_numpy()
        self.processedX = tempProcessedData.drop(forDataLoader.target, axis=1).to_numpy()

    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.processedX, self.processedy, test_size = test_size)
    self.processedData = XAIData(X_train = self.X_train, X_test = self.X_test, y_train = self.y_train, y_test = self.y_test)
      
#Class DataProcessorEnds
