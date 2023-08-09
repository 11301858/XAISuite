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


class DataLoader:
  '''
  Class that loads data from a given source

  :param Union[str, Callable, numpy.ndarray, pd.DataFrame, tuple] data: The data identifier, a function that returns the data, or the data itself in the form of a numpy array or a pandas DataFrame
  :param str, optional source: The source of the data. Either "auto", "system", "preloaded", "generated", or "url". If "auto", the source will be inferred based on `data`. By default, "auto"
  :param str, optional type: The type of data. Either "Tabular", "Image", or "Text". By default, "Tabular". If "Text" or "Image", only one feature is allowed. 
  :param Union[str, list], optional variable_names: The variables in the dataset. By default, set to "auto" and inferred. 
  :param Union[str, list], optional target_names: The target variable(s). By default, set to "auto" and inferred
  :param Union[str, list], optional cut: Variables to drop from the data. By default, None. 
  :param `**dataGenerationArgs`: Additional arguments to pass in if `data` is Callable. 
  :raises ValueError: if data cannot be resolved or if invalid arguments are passed. 
  '''
  def __init__(self, data:Union[str, Callable, numpy.ndarray, pd.DataFrame, tuple], source:str = "auto", type:str = "Tabular", variable_names:Union[str, list] = "auto", target_names:Union[str, list] = "auto", cut:Union[str, list] = None, **dataGenerationArgs):
    '''
    Class constructor
    '''
    self.content = None
    if isinstance(data, Callable):
      data = data(**dataGenerationArgs)



    if isinstance(data, Callable):
      raise ValueError("Callable passed to DataLoader returns another Callable instead of a string, numpy.ndarray, pd.DataFrame, or tuple.")
    
    if isinstance(data, pd.DataFrame):
      self.content = data
    elif isinstance(data, numpy.ndarray):
      self.content = pd.DataFrame(data)
    elif isinstance(data, tuple):
      self.content = pd.DataFrame(data[0])
      self.content["target"] = data[1]
    elif isinstance(data, str):
      match source:
        case "system":
          initializeDataFromSystem(data)
        case "preloaded":
          initializeDataFromPreloaded(data)
        case "generated":
          initializeDataFromGenerated(data, **dataGenerationArgs)
        case "url":
          initializeDataFromUrl(data)
        case "auto":
          try:
            initializeDataFromSystem(data)
          except:
            try:
              initializeDataFromPreloaded(data)
            except:
              try:
                generateArgs = additional.get("dataGenerationArgs")
                generateArgs.update(additional.get("config"))
                initializeDataFromGenerated(data, **generateArgs)
              except:
                try:
                  initializeDataFromUrl(data)
                except:
                  raise ValueError("Data is not valid. Please make sure your data string is not misspelt and exists.")
    assert isinstance(self.content, pd.DataFrame), "A problem occurred with the data loading. If the problem persists, file an issue at github.com/11301858/XAISuite"

    self.content.drop([additional.get("dataTypeArgs").get("cut")], axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise')
    additional.get("dataTypeArgs").pop("cut")

    
    
    
    
                  

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
    :raises NotFoundError: if provided generation configuration is not found
    '''
      
      self.content = pd.DataFrame(eval(id + "(**generateArgs)"))
  
  def initializeDataFromUrl(id:str):
    '''
    Initializes data from URL
  
    :param str id: The url of the data
    '''
    df = pd.read_csv(id)
    self.content = df
    


class DataProcessor:
  '''
  Class that processes data

  :param object, optional processor: The data processer, either a string function, or an object with fit() and transform() methods.
  :param `**processorArgs`: Arguments to be passed into the processor
  '''

  def __init__(processor:object = None, **processorArgs):
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
      self.processor() = processor(**processorArgs)

    


