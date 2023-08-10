class ModelTrainer:
  '''
  Class to train an explainable machine learning model.

  :param object model: The string name of the model, the function returning the model, or the model object itself. The model function must have fit() and predict() functions. A score() function is optional.
  :param DataProcessor withData: The data that will be used to train and test the model
  :param str task: The type of task that the model performs. By default, "Tabular". Other options are "Vision", "NLP", "Data", and "Ranking"
  '''

  def __init__(model:object, withData: Data, task:str = "Tabular", explainers:Union[str, list] = None, **modelArgs):
    tempModel = model
    if isinstance(model, str):
      tempModel = eval(model + "(**modelArgs)")
    elif isinstance(model, Callable):
      tempModel = model(**modelArgs)

    model = tempModel

    try:
      model.fit(withData.X_train, withData.y_train)
    except Exception as e:
      print("Model could not be fit to data: \n" + str(e))

    score = 0

    try:
      score = model.score(withData.X_test, withData.y_test)
    except:
      score = r2_score([model.predict(x) for x in withData.X_test], withData.y_test)

    print("Model score is " + score)
      
    

    
    
    
