
import urllib.request
import gzip
import numpy as np

#Loading the dataset
def load_dataset(url=False, delimeter=","):
    
    """Funciton loads an online .gz dataset with a given delimeter
    
        Parameters
        ----------
        url (optional) : add url to download custom dataset.
        delimeter (optional) : add custom delimeter if ',' is not default for the dataset.
        
        Returns
        -------
        data : Numpy array with the loaded data.
        FEATURES_NAMES : List of names of the attributes for the default dataset.
        TARGET_NAME : List with name of the target for the default dataset.
    """
    try:
        data = np.genfromtxt(gzip.GzipFile(filename="data/covtype.data.gz"), delimiter=",")
        return data
    except:
        try: 
            with urllib.request.urlopen(url) as response:
                with gzip.GzipFile(fileobj=response) as uncompressed:
                    data = np.loadtxt(uncompressed, delimiter=delimeter)
            return data
        except AttributeError:
            print("No dataset file in the folder. Please add file or provide the URL.")

#Names of attributes as in the 'covtype.info' reference file.
FEATURES_NAMES = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]
FEATURES_NAMES += [f"Wilderness_Area_{i}" for i in range(4)]
FEATURES_NAMES += [f"Soil_Type_{i}" for i in range(40)]
TARGET_NAME = ["Cover_Type"]

