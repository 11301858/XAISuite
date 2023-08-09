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

  :param Union[str, Callable, numpy.ndarray, pd.DataFrame] data: The data identifier, a function that returns the data, or the data itself in the form of a numpy array or a pandas DataFrame
  :param str, optional source: The source of the data. Either "auto", "system", "preloaded", "generated", or "url". If "auto", the source will be inferred based on `data`. By default, "auto"
  :param str, optional type: The type of data. Either "Tabular", "Image", or "Text". By default, "Tabular"
  :param dict additional: Additional arguments to pass in. For example, if `data` is Callable, this will house any arguments passed to that function. You can also pass in a value of the variable config here to specify the format of the data returned.
  Ex. dataArgs = {"dataGenerationArgs": {"return_X_y": True}, dataTypeArgs: {"feature_names" : [x1, x2], "target_names": [y1, y2], "cut": x3}, config:"X_y"}
  :raises ValueError: if provided data is not found or if provided config is invalid
  '''
  def __init__(self, data:Union[str, Callable, numpy.ndarray, pd.DataFrame], source:str = "auto", type:str = "Tabular", additional:dict = None):
    if isinstance(data, pd.DataFrame):
      self.content = data
    elif isinstance(data, numpy.ndarray):
      self.content = pd.DataFrame(data)
    elif isinstance(data, Callable):
      if additional is not None and additional.get("dataGenerationArgs") is not None:
        tempContent = data(**additional.get("dataGenerationArgs"))
        if additional.get("config") is not None:
          match additional.get("config"):
            case "complete":
              self.content = pd.DataFrame(tempContent)
            case "X_y":
              self.content = pd.DataFrame(tempContent[0])
              self.content["target"] = tempContent[1]
            case _:
              raise ValueError("Config value of data is invalid. Must be either 'complete' or 'X_y'")  
        else:
          self.content = pd.DataFrame(tempContent)
    elif isinstance(data, str):
      match source:
        case "system":
          initializeDataFromSystem(data)
        case "preloaded":
          initializeDataFromPreloaded(data)
        case "generated":
          generateArgs = additional.get("dataGenerationArgs")
          generateArgs.update(additional.get("config"))
          initializeDataFromGenerated(data, **generateArgs)
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

    self.content = eval(type + "(**additional.get('dataTypeArgs'))")
                  

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
    if generateArgs is not None and generateArgs.get("config") is not None:
      configuration = generateArgs.get("config")
      generateArgs.pop("config")
      match configuration:
        case "complete":
          data = eval(id + "(" + generateArgs + ")")
          assert isinstance(data, numpy.ndarray), "Command does not return a numpy ndarray."
          self.content = pd.DataFrame(data)
        case "X_y":
          self.content = pd.DataFrame(data[0])
          self.content["target"] = data[1]
          assert isinstance(data[0], numpy.ndarray), "Command does not return a numpy ndarray."
          assert isinstance(data[1], numpy.ndarray), "Command does not return a numpy ndarray."
        case _:
          raise NotFoundError("Configuration " + configuration + " not found.")
    else:
      data = eval(id + "(" + generateArgs + ")")
      assert isinstance(data, numpy.ndarray), "Command does not return a numpy ndarray."
      self.content = pd.DataFrame(data)
  
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
  '''



