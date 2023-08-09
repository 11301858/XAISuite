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
  :param `**dataArgs`: Additional arguments to pass in. For example, if `data` is Callable, this will house any arguments passed to that function. 
  '''
  def __init__(self, data:Union[str, Callable, numpy.ndarray, pd.DataFrame], source:str = "auto", type:str = "Tabular", **dataArgs):
    


class DataProcessor:
  '''
  Class that processes data
  '''



