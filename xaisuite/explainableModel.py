class ModelTrainer:
  '''
  Class to train an explainable machine learning model.

  :param object model: The string name of the model, or the model function itself. The model function must have fit() and predict() functions. A score() function is optional.
  :param DataProcessor withDataProcessor: The data processor 
  '''

  def __init__(model:object, withDataProcessor: DataProcessor, task:str = "Tabular", explainers:Union[str, list] = None, **modelArgs):
    
