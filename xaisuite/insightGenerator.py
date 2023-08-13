class InsightGenerator:
  '''
  Class to generate insights based on explanation results
  '''
  def __init__(explanations:collections.OrderedDict):
    explainers = list(explanations.keys())
    explainers.remove('predict')
    for explainer in explainers:
      temp = explanations.get(explainer)

  def getShreyanDistance(vec1:list, vec2:list):
    '''
    Calculate the distance between two ordered vectors

    :param list vec1: The pattern vector
    :param list vec2: The disorder vector
    '''

    
    
    
      

