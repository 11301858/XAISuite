from .xaisuiteFoundation import*
from .dataHandler import DataProcessor

class ModelTrainer:
  '''
  Class to train an explainable machine learning model.
  
  :param Any model: The string name of the model, the function returning the model, or the model Any itself. The model function must have fit() and predict() functions. A score() function is optional.
  :param DataProcessor withData: The data that will be used to train and test the model
  :param str, optional taskType: The type of task that the model performs. By default, "Tabular". Other options are "Vision" and "NLP"
  :param str, optional task: The task that the model performs. By default, "regression". Other option is "classification"
  :param list, dict, optional explainers: A list of explainer names or, if specific parameters need to be passed to the explainers, a dict that contains the explainer names and explainer arguments. Ex. explainers = ["lime", "shap", "mace"] or explainers = {"lime": {"kernel_width": 3}, "shap": {"nsamples": 100}, "mace": None}
  '''

  def __init__(self, model:Any, withData: DataProcessor, taskType:str = "Tabular", task:str = "regression", explainers:Union[list, dict] = None, **modelArgs):
    '''
    Class constructor
    '''
    self.withData = withData
    tempModel = model
    if isinstance(model, str):
      try:
        tempModel = eval(model + "(**modelArgs)")
      except:
        if model in linkModels.keys():
          exec("from " + linkModels.get(model) + " import*")
          tempModel = eval(model + "(**modelArgs)")
    elif isinstance(model, Callable):
        tempModel = model(**modelArgs)

    model = tempModel
    self.model = model

    try:
        model.fit(withData.X_train, withData.y_train)
    except Exception as e:
        print("Model could not be fit to data: \n" + str(e))

    score = 0

    try:
        score = model.score(withData.X_test, withData.y_test)
    except:
        score = r2_score([model.predict(x) for x in withData.X_test], withData.y_test)

    print("Model score is " + str(score))

    #Model has been trained by this point. Now for the explanations

    if explainers is None:
      return

    explainer_names = explainers if isinstance(explainers, list) else explainers.keys()

    self.explainer = eval(taskType + "Explainer(explainers = explainer_names, mode = task, data = withData.loader.wrappedData, model = model, preprocess = withData.processor.transform, params = explainers if isinstance(explainers, dict) else None)")
    self.requestedExplanations = None

  def getExplanationsFor(self, testIndex:Union[int, list] = None, feature_values:dict = None) -> dict:
    '''
    Function to get the local explanations for a particular testing instance. 
    
    :param Union[int, list], optional testIndex: The indices of the testing data for which to fetch local explanations. If empty, local explanations for all instances are returned. If None, `feature_values` is used.
    :param dict, optional feature_values: The values of the features corresponding to a particular index. If None, `testIndex` is used. 
    :returns dict explanations: The requested explanations
    :raises ValueError: If neither testIndex or feature_values is passed
    '''
    if testIndex is None and feature_values is None:
        raise ValueError("One of testIndex or feature_values must be provided.")

    if testIndex is not None and feature_values is not None:
        print("Both testIndex and feature_values were provided. Using testIndex.")

    print("Generating explanations.")

    if isinstance(testIndex, list) and len(testIndex) == 0:
        self.requestedExplanations = self.explainer.explain(self.withData.processor.invert(self.withData.processedData.X_test))
        return self.requestedExplanations
    elif isinstance(testIndex, list) and len(testIndex) >0:
        self.requestedExplanations = self.explainer.explain(self.withData.processor.invert(numpy.array([self.withData.processedData.X_test[i] for i in testIndex])))
        return self.requestedExplanations
    elif isinstance(testIndex, int):
        self.requestedExplanations = self.explainer.explain(self.withData.processor.invert(numpy.array(self.withData.processedData.X_test[testIndex])))
        return self.requestedExplanations

  #By this point, we are aware that we are dealing with feature_values and testIndex must be None. 

    assert(testIndex is None), "Something went wrong. Please try again."

  #First, we fetch all the feature data for the dataset
    data = self.withData.loader.X
  
  #Next, we create a string that resolves into a boolean expression when evaluated and will be a search query constructed from the provided feature_values
    query = []
    for feature, value in feature_values.items():
        query.append("df['" + feature + "'] == " + value)
    queryString = " and ".join(query)
    self.requestedExplanations = self.explainer.explain(data.loc[eval(queryString)])
    return self.requestedExplanations

  def getSummaryExplanations(self) -> dict:
    '''
    Returns global explanations
    
    :returns dict explanations: The requested global explanations
    '''
    self.requestedExplanations = self.explainer.explain_global()
    return self.requestedExplanations

  def getAllExplanations(self) -> dict:
    return self.getExplanationsFor([])

  def plotExplanations(self, explainer:str = None, index:int = 0):
    '''
    Plot explanations

    :param str explainer: The explainer for which to plot explanations
    :param int index: The instance for which to plot explanations, in the numerical order returned by the explanation retrieval function. By default, 0. 
    '''
    try:
      self.requestedExplanations[explainer].ipython_plot(index)
    except Exception as e:
      print("Plotting explanations failed. Make sure you call getExplanationsFor, getAllExplanations, or getSummaryExplanations before plotting.")
      print("Error statement: " + str(e))
      
    
