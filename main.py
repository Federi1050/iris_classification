from it.valtellina.machine_learning.dataset_manager import DatasetManager

testing = True

if testing:
    pass
    dataset_mg = DatasetManager()
    #print("prima")
    #print(dataset_mg.get_data().head())
    dataset_mg.split_data('Species')
    #print(dataset_mg.get_X_train().head())
    dataset_mg.scaling()
    


else:
    pass
    # flask