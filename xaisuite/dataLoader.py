from .imports import*


def load_data_CSV(data:str, target:str, cut: Union[str, list] = None) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular dataset.
    Parameters:
    data:str | Pathname for the CSV file where the dataset is found.
    target:str | Target variable used for training data
    cut: Union[str, list] = None | Variables that should be ignored in training
    
    Returns:
    tabular_data: Tabular | Tabular object instance representing 'data'
    '''
    try:
        df = pd.read_csv(data) #We read the dataset onto a dataframes object
        if cut is not None:
          df.drop(cut, axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
          print(df)
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object

def load_data_sklearn(datastore:dict, target:str, cut: Union[str, list] = None) -> Tabular: # Returns tabular data
    '''A function that creates a omnixai.data.tabular.Tabular object instance representing a particular sklearn dataset for demoing.
    Parameters:
    datastore:dict | A dictionary object containing the data
    target:str | Target variable used for training data
    cut: Union[str, list] = None | Variables that should be ignored in training
    
    Returns:
    tabular_data: Tabular | Tabular object instance representing 'data'
    '''
    try:
        df = pd.DataFrame(data=datastore.data, columns=datastore.feature_names) #We read the dataset (exluding target) onto a dataframes object
        df["target"] = datastore.target #We add the target column
        if cut is not None:
          df.drop([cut], axis = 1, index = None, columns = None, level = None, inplace = True, errors = 'raise') #Remove columns the user doesn't want to include
        tabular_data = Tabular(df, target_column=target) #Create a Tabular object (needed for future training and explaining) and specify the target column. Omnixai does not allow multiple targets, so datasets containing 2 or more targets need to be passed twice through the program, with one of the targets being cut. 
        return tabular_data #Return data through a Tabular object
    except:
        raise ValueError("Unable to load data properly. Make sure your file is in the same directory and that the target is present") #This means that there was a problem reading the data, cutting columns, or creating the Tabular object

