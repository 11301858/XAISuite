class InsightGenerator:
  '''
  Class to generate insights based on explanation results
  '''
  def __init__(explanations:collections.OrderedDict):
    explainers = list(explanations.keys())
    explainers.remove('predict')
    for explainer in explainers:
      temp = explanations.get(explainer)
      

