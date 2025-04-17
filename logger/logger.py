import io
import json
import numpy as np

class Logger:
    def __init__(self, directory):
        self.directory = directory



    def save_results(self, **kwargs):
        """
        Save results to a JSON file.
        
        Args:
            filename (str): Name of the file to save results to
            **kwargs: Arbitrary keyword arguments containing results to save
        """

        # Save the json
        results = {}
    
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                results[key] = [arr.tolist() for arr in value]
            else:
                results[key] = value
        
        name = self.directory + "/pg_results2.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()