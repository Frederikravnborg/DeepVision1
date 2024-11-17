import os
import pickle

# Configuration for file paths
DATASET_DIR = 'Potholes/'
TRAINING_DATA_FILE = os.path.join(DATASET_DIR, 'training_data.pkl')

def inspect_training_data(training_data_file):
    try:
        # Load the training data from the pickle file
        with open(training_data_file, 'rb') as f:
            combined_data = pickle.load(f)
        
        # Debugging the format of the loaded data
        print("Successfully loaded the training data.\n")
        
        # Print the top-level structure of the loaded data
        print("Keys in the training data:", combined_data.keys())
        
        # Extract the proposals and ground_truths
        proposals = combined_data.get('proposals', [])
        ground_truths = combined_data.get('ground_truths', [])
        
        print(f"\nTotal Proposals Loaded: {len(proposals)}")
        print(f"Total Ground Truths Loaded: {len(ground_truths)}")

        # Debugging the format of proposals and their bounding boxes
        for i, proposal in enumerate(proposals[:5]):  # Print the first 5 proposals for inspection
            print(f"\nProposal {i+1}:")
            print("Proposal Data:", proposal)
            print("Bounding Box (bbox):", proposal.get('bbox'))
            print("Label:", proposal.get('label'))
            
            # Check the bbox format (should be a dictionary with xmin, ymin, xmax, ymax)
            bbox = proposal.get('bbox')
            if bbox:
                print(f"Valid bbox check: {len(bbox)} keys")
                try:
                    # Accessing individual values and converting them to float
                    xmin = float(bbox['xmin'])
                    ymin = float(bbox['ymin'])
                    xmax = float(bbox['xmax'])
                    ymax = float(bbox['ymax'])
                    print(f"Converted bbox: ({xmin}, {ymin}, {xmax}, {ymax})")
                except ValueError as e:
                    print(f"Error converting bbox values to float: {e}")
                except KeyError as e:
                    print(f"Missing key in bbox: {e}")
            else:
                print("Invalid bbox: No bbox found")
        
        # Optionally, you can inspect a few ground_truth entries similarly
        print("\nInspecting first 3 ground truth entries:")
        for i, ground_truth in enumerate(ground_truths[:3]):
            print(f"\nGround Truth {i+1}:")
            print("Ground Truth Data:", ground_truth)
            print("Bounding Box (bbox):", ground_truth.get('bbox'))
            print("Label:", ground_truth.get('label'))
            
    except Exception as e:
        print(f"Error loading or inspecting the data: {e}")

# Run the function to inspect the training data
inspect_training_data(TRAINING_DATA_FILE)
