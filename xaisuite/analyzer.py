from .imports import*

def compare_explanations(filenames:list): #Analyze the generated explanations in given files
  '''A function that analyzes and compares the explanations generated by train_and_explainModel.
    Parameters:
    filenames:list | File names with explanations (of the form "Explainer ImportanceScores - Model Target.csv")
    
    Returns:
    Nothing
    '''
  explainers = []
  features = [] #All files should have same features
  for filename in filenames:
      try: 
        df = pd.read_csv(filename)

        for i in range(len(df['features'])):
            df['features'][i] = ast.literal_eval(df['features'][i])

        for i in range(len(df['scores'])):
            df['scores'][i] = ast.literal_eval(df['scores'][i])

        features = df['features']
        scores = df['scores']
        explainer = filename.split()[0]
        explainers.append(explainer)

        for feature in features[0]:
            vars()[feature + explainer + "List"] = []
            for i in range(len(df['features'])):
                eval(feature + explainer + "List").append(scores[i][features[i].index(feature)])

        vars()[explainer + "maxScore"] = []
        for score in scores:
            eval(explainer + "maxScore").append(max(score))   
      except:
        print("An error occurred while analyzing the graph.")

  for feature in features[0]:
      for explainer in explainers:
          plt.plot(eval(feature + explainer + "List"))
          plt.xlabel("Instance #")
          plt.ylabel("Importance Score")
      plt.title("Change in importance of " + feature + " over instance number " + "- " + ' '.join([str(elem) for elem in explainers]))
      plt.show()

  for explainer in explainers:
      plt.plot(eval(explainer + "maxScore"))
      plt.xlabel("Instance #")
      plt.ylabel("Importance Score")
  plt.title("Change in importance of most important feature over instance number " + "- " + ' '.join([str(elem) for elem in explainers]))
  plt.show()  
  

  for feature in features[0]:
    feature_explainer_lists = []
    for explainer in explainers:
        feature_explainer_lists.append(eval(feature + explainer + "List"))
    correlation = np.corrcoef([x for x in feature_explainer_lists])
    if(len(explainers) == 2):
        print ("Correlation between " + ' and '.join([str(elem) for elem in explainers]) + " for feature " + feature + ": " + str(correlation[1,0]))
    else:
        print (correlation)
