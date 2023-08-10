class ModelTrainer:
  '''
  Class to train an explainable machine learning model.

  :param object model: The string name of the model, or the model function itself. The model function must have fit() and predict() functions. A score() function is optional.
  :param DataProcessor withData: The data that will be used to train and test the model
  :param str task: The type of task that the model performs. By default, "Tabular". Other options are "Vision", "NLP", "Data", and "Ranking"
  '''

  def __init__(model:object, withData: Data, task:str = "Tabular", explainers:Union[str, list] = None, **modelArgs):
    
    
