class Data:
  '''
  Class representing a dataset
  '''

 


class UnprocessedData(Data):
  '''
  Class representing an unprocessed dataset that inherits from Data
  '''
  

class ProcessedData(Data):
  '''
  Class representing a processed dataset that inherits from Data
  '''

#Class DataLoader starts here

class DataLoader:
  '''
  Class that loads data from a given source

  :param Union[str, Callable, numpy.ndarray, pd.DataFrame, tuple] data: The data identifier, a function that returns the data, or the data itself in the form of a numpy array, pandas DataFrame, or tuple
  :param str, optional source: The source of the data. Either "auto", "system", "preloaded", "generated", or "url". If "auto", the source will be inferred based on `data`. By default, "auto"
  :param str, optional type: The type of data. Either "Tabular", "Image", or "Text". By default, "Tabular". If "Text" or "Image", only one feature is allowed. 
  :param Union[str, list], optional variable_names: The variables in the dataset excluding `cut`, in the order that they appear in the data. By default, set to "auto" and inferred. 
  :param Union[str, list], optional target_names: The target variable(s). By default, set to "auto" and inferred
  :param Union[str, list], optional cut: Variables to drop from the data. By default, None. 
  :param Union[str, list], optional categorical: If type == "Tabular", variables that contain categorical data.
  :param `**dataGenerationArgs`: Additional arguments to pass in if `data` is Callable. 
  :raises ValueError: if data cannot be resolved or if invalid arguments are passed. 
  '''
  def __init__(self, data:Union[str, Callable, numpy.ndarray, pd.DataFrame, tuple], source:str = "auto", type:str = "Tabular", variable_names:Union[str, list] = "auto", target_names:Union[str, list] = "auto", cut:Union[str, list] = None, categorical:Union[str, list] = None, **dataGenerationArgs):
    '''
    Class constructor
    '''
    #Initially, we set the data content to None. self.content is meant to be a pd.DataFrame
    self.type = type
    self.content = None
    #If data is a function that returns the dataset or a string representation of such a function, we reassign data to the return value of this function. 
    if isinstance(data, Callable):
      data = data(**dataGenerationArgs)


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
          initializeDataFromSystem(data) #The source is system, so we search for a file
        case "preloaded":
          initializeDataFromPreloaded(data) #The source is preloaded, so we search for the data in a preloaded dictionary
        case "generated":
          initializeDataFromGenerated(data, **dataGenerationArgs) #The source is generated, and the string is the name of a function, so we pass data and the variables to pass to the data-generating function
        case "url":
          initializeDataFromUrl(data) #The source is a url, so we try to get the data from the url
        case "auto": #Here, the user is either unaware of where to look or just lazy :) Anyway, we need to do some hard work. We need to try each data loading method and see what sticks.
          try:
            initializeDataFromSystem(data) 
          except:
            try:
              initializeDataFromPreloaded(data)
            except:
              try:
                initializeDataFromGenerated(data, **dataGenerationArgs)
              except:
                try:
                  initializeDataFromUrl(data)
                except:
                  raise ValueError("Data is not valid. Please make sure your data string is not misspelt and exists.") #This means we could not find the data anywhere.
                  
    assert isinstance(self.content, pd.DataFrame), "A problem occurred with the data loading. If the problem persists, file an issue at github.com/11301858/XAISuite" #By this time, data should definitely be a dataframe. If it is not, something has gone horribly wrong. 

    if cut is not None:
      self.content.drop(cut, axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove the cut variable (if it exists)

    #Now, we should have a dataframe with the features and the target. If the data type is image or text, there should be only one feature

    if (type == "Image" or type == "Text") and len(variable_names) < 2:
      return ValueError("For Image and Text data, there must be only one feature.")

    #Make sure that variable_names are the same length as the number of columns in the data provided. 

    if (len(variable_names) != len(self.content.columns)):
      raise ValueError("The length of variable_names is incompatible with the data provided.")

    #Make sure that the target variable, if provided, is in the provided variable_names, if provided

    if target != "auto" && variable_names != "auto" && target not in variable_names:
      raise ValueError("Custom target variable name not found in variable_names.")

    #If the target variable is auto, we determine what the value of the target variable will be first:
    if target == "auto" and "target" in self.content.columns:
      target = "target"
    elif target == "auto" and "target" not in self.content.columns:
      target = self.content.columns[-1]

    #If variable_names is auto and target is not auto, there is nothing that needs to be done

    

    #If variable_names is not auto and target is auto, we need to make sure that variable_names does not override the auto target value. 
    if variable_names != "auto" and target == "auto":
      target = variable_names[-1] if not isinstance(variable_names, str) else variable_names
    else if variable_names != "auto" and target == "target":
      target = variable_names[self.content.columns.get_loc("target")]

    #Now we set the variable_names.
    if variable_names != "auto": #Do nothing if the variable_names are set to auto
      self.content.columns = variable_names

    if categorical not in variable_names:
      raise ValueError("Categorical variables provided are incompatible with variable_names.")

    #Now we split the data to make it easier to handle:

    self.y = self.content[target]
    self.X = self.content.drop("target")

    #Now we're ready to finalize creating the data object
    self.wrappedData = None
    if type == "Tabular":
      self.wrappedData = Tabular(data = self.content, feature_columns = self.x.columns, categorical_columns = categorical, target_column = self.y.columns)
    else if type == "Image":
      self.wrappedData = Image(data = self.x, batched = True)
    else if type == "Text":
      self.wrappedData = Text(data = self.x.values.reshape(-1,).tolist())

  def initializeDataFromSystem(id:str):
    '''
    Initializes data from user's system
  
    :param str id: The data file path
    :raises NotFoundError: if provided file path is not found or is not a file
    '''
    if not os.path.isfile(id):
      raise NotFoundError("Given data id is not a file path.")
    
    df = pd.read_csv(id)
    self.content = df
    
  
  def initializeDataFromPreloaded(id:str):
    '''
    Initializes data from preloaded sklearn datasets
  
    :param str id: The name of the preloaded dataset
    :raises NotFoundError: if provided preloaded data name is not found
    '''
    if id in acceptedDataIDs:
      data = eval(acceptedDataIDs.get(id) + "(return_X_y=True)")
      self.content = pd.DataFrame(data[0])
      self.content["target"] = data[1]
    else:
      raise NotFoundError("Not an accepted preloaded data id. Available preloaded data ids are: \n" + acceptedDataIDs.keys() + "\n")
  
  def initializeDataFromGenerated(id:str, **generateArgs):
    '''
    Initializes data from string commands
  
    :param str id: The string generation command
    :param `**generateArgs`: Arguments to pass to the function represented by `id`
    :raises NotFoundError: if provided generation configuration is not found
    '''
      data = eval(id + "(**generateArgs)")
      if isinstance(data, tuple): #We take casre of the case that the generatorfunction returns a tuple
        temp = pd.DataFrame(data[0])
        temp["target"] = data[1]
        self.content = temp
      else:
        self.content = pd.DataFrame(data)

      
        
  
  def initializeDataFromUrl(id:str):
    '''
    Initializes data from URL
  
    :param str id: The url of the data
    '''
    df = pd.read_csv(id)
    self.content = df
    
#Class DataLoader ends here

class DataProcessor:
  '''
  Class that processes data
  :param DataLoader forDataLoader: The dataloader that will be associated with this processor. 
  :param float, optional test_size: The proportion of data that will be used to test and score the machine learning model. By default, 0.2
  :param object, optional processor: The data processer, either a string function, or an object with fit() and transform() methods.
  :param `**processorArgs`: Arguments to be passed into the processor
  '''

  def __init__(forDataLoader:DataLoader, test_size = 0.2:float, processor:object = None, **processorArgs):
    self.processor = processor
    compositeTabularProcessorArgs = None
    
    if processor == "TabularTransform" and processorArgs is not None:
      for component in processorArgs.items():
        if isinstance(component[1], str):
          compositeTabularProcessorArgs.update((component[0], eval(component[1] + "()")))
        else:
          compositeTabularProcessorArgs.update((component[0], component[1]))
      processorArgs = compositeTabularProcessorArgs
        
    if isinstance(processor, str):
      self.processor = self.eval(processor + "(**processorArgs)")
    elif isinstance(processor, Callable):
      self.processor = processor(**processorArgs)

  

    self.processor.fit(forDataLoader.wrappedData)

    

