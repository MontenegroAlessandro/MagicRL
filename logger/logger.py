import io
import json
import numpy as np
import time
#import tensorflow as tf

class Logger:
    def __init__(self, directory):
        self.directory = directory
        self.start_time = time.time()
        #self.summary_writer = tf.summary.create_file_writer(self.directory + "/tensorboard")



    def save_results(self, **kwargs):
        """
        Save results to a JSON file.
        
        Args:
            filename (str): Name of the file to save results to
            **kwargs: Arbitrary keyword arguments containing results to save
        """
        # Save the json
        results = {}
        results["time"] = time.time() - self.start_time
    
        for key, value in kwargs.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
            elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                results[key] = [arr.tolist() for arr in value]
            else:
                results[key] = value

        
        name = self.directory + "/pg_results.json"
        with io.open(name, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False, indent=4))
            f.close()

    def save_tensorboard_results(self, step=None, **kwargs):
        """
        Log results to TensorBoard.
        
        Args:
            step (int, optional): The global step value to record with the summary.
                                If None, will use time since start as step.
            **kwargs: Arbitrary keyword arguments containing results to log to TensorBoard
        """
        
        # If step is not provided, use time since start
        if step is None:
            step = int(time.time() - self.start_time)
        
        
        # Process values for TensorBoard
        with self.summary_writer.as_default():
            for key, value in kwargs.items():
                # Handle numpy arrays
                if isinstance(value, np.ndarray):
                    if value.size == 1:  # Scalar
                        tf.summary.scalar(key, float(value.item()), step=step)
                    else:  # Non-scalar array
                        tf.summary.histogram(key, value, step=step)
                
                # Handle lists of numpy arrays
                elif isinstance(value, list) and len(value) > 0 and isinstance(value[0], np.ndarray):
                    # Log each array separately
                    for i, arr in enumerate(value):
                        sub_key = f"{key}_{i}"
                        if arr.size == 1:  # Scalar
                            tf.summary.scalar(sub_key, float(arr.item()), step=step)
                        else:  # Non-scalar array
                            tf.summary.histogram(sub_key, arr, step=step)
                
                # Handle scalar values
                elif isinstance(value, (int, float)):
                    tf.summary.scalar(key, value, step=step)
                
                # For everything else, try to log as scalar, fallback to text
                else:
                    try:
                        # Try to convert to scalar
                        scalar_value = float(value)
                        tf.summary.scalar(key, scalar_value, step=step)
                    except (ValueError, TypeError):
                        # If not convertible to float, log as text
                        tf.summary.text(key, str(value), step=step)
        
        # Flush to ensure data is written
        self.summary_writer.flush()

            