class ModelTrainer:
  '''
  Class to train an explainable machine learning model.

  :param Any model: The string name of the model, the function returning the model, or the model Any itself. The model function must have fit() and predict() functions. A score() function is optional.
  :param DataProcessor withData: The data that will be used to train and test the model
  :param str, optional taskType: The type of task that the model performs. By default, "Tabular". Other options are "Vision" and "NLP"
  :param str, optional task: The task that the model performs. By default, "regression". Other option is "classification"
  :param list, dict, optional explainers: A list of explainer names or, if specific parameters need to be passed to the explainers, a dict that contains the explainer names and explainer arguments. 
  Ex. explainers = ["lime", "shap", "mace"] or explainers = {"lime": {"kernel_width": 3}, "shap": {"nsamples": 100}, "mace": None}
  '''

  def __init__(model:Any, withData: DataProcessor, taskType:str = "Tabular", task:str = "regression", explainers:Union[list, dict] = None, **modelArgs):
    '''
    Class constructor
    '''
    tempModel = model
    if isinstance(model, str):
      tempModel = eval(model + "(**modelArgs)")
    elif isinstance(model, Callable):
      tempModel = model(**modelArgs)

    model = tempModel

    try:
      model.fit(withData.X_train, withDataLoader.y_train)
    except Exception as e:
      print("Model could not be fit to data: \n" + str(e))

    score = 0

    try:
      score = model.score(withData.X_test, withData.y_test)
    except:
      score = r2_score([model.predict(x) for x in withData.X_test], withData.y_test)

    print("Model score is " + score)

    #Model has been trained by this point. Now for the explanations

    explainer_names = explainers if isinstance(explainers, list) else explainers.keys()

    explainer = eval(taskType + "Explainer(explainers = explainer_names, mode = task, data = withData.loader.wrappedData, preprocess = withData.processor, params = explainers if isinstance(explainers, dict) else None)")
    
    

    
    
    
