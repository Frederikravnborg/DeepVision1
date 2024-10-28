import matplotlib.pyplot as plt
import numpy as np

# Define the data
models = ['EncDec', 'UNet', 'UNet2', 'DilatedNet']
datasets = ['Training', 'Validation', 'Test']
metrics = ['Dice Coefficient', 'IoU', 'Accuracy', 'Sensitivity', 'Specificity']

data_phc_bce = {
    'EncDec': {
        'Training': {
            'Dice Coefficient': 0.9809,
            'IoU': 0.8980,
            'Accuracy': 0.9790,
            'Sensitivity': 0.9654,
            'Specificity': 0.9841
        },
        'Validation': {
            'Dice Coefficient': 0.9734,
            'IoU': 0.8543,
            'Accuracy': 0.9660,
            'Sensitivity': 0.9444,
            'Specificity': 0.9737
        },
        'Test': {
            'Dice Coefficient': 0.9736,
            'IoU': 0.8632,
            'Accuracy': 0.9659,
            'Sensitivity': 0.9486,
            'Specificity': 0.9724
        }
    },
    'UNet': {
        'Training': {
            'Dice Coefficient': 0.9934,
            'IoU': 0.9637,
            'Accuracy': 0.9929,
            'Sensitivity': 0.9856,
            'Specificity': 0.9957
        },
        'Validation': {
            'Dice Coefficient': 0.9912,
            'IoU': 0.9485,
            'Accuracy': 0.9891,
            'Sensitivity': 0.9750,
            'Specificity': 0.9938
        },
        'Test': {
            'Dice Coefficient': 0.9901,
            'IoU': 0.9459,
            'Accuracy': 0.9872,
            'Sensitivity': 0.9768,
            'Specificity': 0.9912
        }
    },
    'UNet2': {
        'Training': {
            'Dice Coefficient': 0.9950,
            'IoU': 0.9722,
            'Accuracy': 0.9946,
            'Sensitivity': 0.9872,
            'Specificity': 0.9974
        },
        'Validation': {
            'Dice Coefficient': 0.9889,
            'IoU': 0.9385,
            'Accuracy': 0.9845,
            'Sensitivity': 0.9712,
            'Specificity': 0.9895
        },
        'Test': {
            'Dice Coefficient': 0.9901,
            'IoU': 0.9458,
            'Accuracy': 0.9862,
            'Sensitivity': 0.9755,
            'Specificity': 0.9903
        }
    },
    'DilatedNet': {
        'Training': {
            'Dice Coefficient': 0.9896,
            'IoU': 0.9433,
            'Accuracy': 0.9892,
            'Sensitivity': 0.9806,
            'Specificity': 0.9925
        },
        'Validation': {
            'Dice Coefficient': 0.9849,
            'IoU': 0.9172,
            'Accuracy': 0.9815,
            'Sensitivity': 0.9742,
            'Specificity': 0.9842
        },
        'Test': {
            'Dice Coefficient': 0.9842,
            'IoU': 0.9154,
            'Accuracy': 0.9800,
            'Sensitivity': 0.9682,
            'Specificity': 0.9846
        }
    }
}

data_ph2_bce = {
    'EncDec': {
        'Training': {
            'Dice Coefficient': 0.9833,
            'IoU': 0.9150,
            'Accuracy': 0.9809,
            'Sensitivity': 0.9678,
            'Specificity': 0.9863
        },
        'Validation': {
            'Dice Coefficient': 0.9519,
            'IoU': 0.8016,
            'Accuracy': 0.9164,
            'Sensitivity': 0.8698,
            'Specificity': 0.9570
        },
        'Test': {
            'Dice Coefficient': 0.9649,
            'IoU': 0.8212,
            'Accuracy': 0.9482,
            'Sensitivity': 0.8974,
            'Specificity': 0.9687
        }
    },
    'UNet': {
        'Training': {
            'Dice Coefficient': 0.9813,
            'IoU': 0.9043,
            'Accuracy': 0.9791,
            'Sensitivity': 0.9619,
            'Specificity': 0.9860
        },
        'Validation': {
            'Dice Coefficient': 0.9734,
            'IoU': 0.8484,
            'Accuracy': 0.9680,
            'Sensitivity': 0.9463,
            'Specificity': 0.9791
        },
        'Test': {
            'Dice Coefficient': 0.9719,
            'IoU': 0.8537,
            'Accuracy': 0.9626,
            'Sensitivity': 0.9219,
            'Specificity': 0.9835
        }
    },
    'UNet2': {
        'Training': {
            'Dice Coefficient': 0.9826,
            'IoU': 0.9097,
            'Accuracy': 0.9779,
            'Sensitivity': 0.9716,
            'Specificity': 0.9788
        },
        'Validation': {
            'Dice Coefficient': 0.9602,
            'IoU': 0.8281,
            'Accuracy': 0.9364,
            'Sensitivity': 0.9467,
            'Specificity': 0.9261
        },
        'Test': {
            'Dice Coefficient': 0.9620,
            'IoU': 0.8263,
            'Accuracy': 0.9414,
            'Sensitivity': 0.9350,
            'Specificity': 0.9415
        }
    },
    'DilatedNet': {
        'Training': {
            'Dice Coefficient': 0.9789,
            'IoU': 0.8936,
            'Accuracy': 0.9766,
            'Sensitivity': 0.9581,
            'Specificity': 0.9845
        },
        'Validation': {
            'Dice Coefficient': 0.9635,
            'IoU': 0.8059,
            'Accuracy': 0.9482,
            'Sensitivity': 0.8990,
            'Specificity': 0.9579
        },
        'Test': {
            'Dice Coefficient': 0.9640,
            'IoU': 0.8129,
            'Accuracy': 0.9533,
            'Sensitivity': 0.9324,
            'Specificity': 0.9618
        }
    }
}

data_drive_bce = {
    'EncDec': {
        'Training': {
            'Dice Coefficient': 0.8864,
            'IoU': 0.0605,
            'Accuracy': 0.8721,
            'Sensitivity': 0.0000,
            'Specificity': 1.0000
        },
        'Validation': {
            'Dice Coefficient': 0.8902,
            'IoU': 0.0599,
            'Accuracy': 0.8817,
            'Sensitivity': 0.0000,
            'Specificity': 1.0000
        },
        'Test': {
            'Dice Coefficient': 0.8817,
            'IoU': 0.0614,
            'Accuracy': 0.8550,
            'Sensitivity': 0.0000,
            'Specificity': 1.0000
        }
    },
    'UNet': {
        'Training': {
            'Dice Coefficient': 0.9299,
            'IoU': 0.3199,
            'Accuracy': 0.8989,
            'Sensitivity': 0.6146,
            'Specificity': 0.9969
        },
        'Validation': {
            'Dice Coefficient': 0.9343,
            'IoU': 0.3304,
            'Accuracy': 0.8884,
            'Sensitivity': 0.5366,
            'Specificity': 0.9993
        },
        'Test': {
            'Dice Coefficient': 0.9195,
            'IoU': 0.3106,
            'Accuracy': 0.8816,
            'Sensitivity': 0.6185,
            'Specificity': 0.9819
        }
    },
    'UNet2': {
        'Training': {
            'Dice Coefficient': 0.9281,
            'IoU': 0.3776,
            'Accuracy': 0.9004,
            'Sensitivity': 0.8110,
            'Specificity': 0.9914
        },
        'Validation': {
            'Dice Coefficient': 0.9281,
            'IoU': 0.3466,
            'Accuracy': 0.9094,
            'Sensitivity': 0.8249,
            'Specificity': 0.9873
        },
        'Test': {
            'Dice Coefficient': 0.9311,
            'IoU': 0.3671,
            'Accuracy': 0.9058,
            'Sensitivity': 0.7654,
            'Specificity': 0.9937
        }
    },
    'DilatedNet': {
        'Training': {
            'Dice Coefficient': 0.8789,
            'IoU': 0.1613,
            'Accuracy': 0.8789,
            'Sensitivity': 0.3031,
            'Specificity': 0.9909
        },
        'Validation': {
            'Dice Coefficient': 0.8767,
            'IoU': 0.1374,
            'Accuracy': 0.8687,
            'Sensitivity': 0.0843,
            'Specificity': 0.9978
        },
        'Test': {
            'Dice Coefficient': 0.8781,
            'IoU': 0.1586,
            'Accuracy': 0.8736,
            'Sensitivity': 0.2040,
            'Specificity': 0.9974
        }
    }
}


data = data_drive_bce  # Choose the dataset to plot

# Plotting
import matplotlib.pyplot as plt
import numpy as np

# Set up the figure and axes
fig, axs = plt.subplots(1, len(metrics), figsize=(20, 5), sharey=False)

# Colors for datasets
colors = {'Training': 'skyblue', 'Validation': 'lightgreen', 'Test': 'lightcoral'}
dataset_labels = list(colors.keys())

# Iterate over the metrics and plot
for idx, metric in enumerate(metrics):
    ax = axs[idx]
    x = np.arange(len(models))  # the label locations
    width = 0.2  # the width of the bars

    for i, dataset in enumerate(datasets):
        metric_values = [data[model][dataset][metric] for model in models]
        offset = (i - 1) * width  # Adjust the position of the bar
        ax.bar(x + offset, metric_values, width, label=dataset if idx == 0 else "", color=colors[dataset])

    # Add some text for labels, title, and custom x-axis tick labels, etc.
    ax.set_ylabel(metric)
    ax.set_title(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(models)

    # Rotate x labels if needed
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


    # Collect all metric values for the current metric across all models and datasets
    all_metric_values = []
    for dataset in datasets:
        metric_values = [data[model][dataset][metric] for model in models]
        all_metric_values.extend(metric_values)

    # Calculate the minimum and maximum values
    min_val = min(all_metric_values)
    max_val = max(all_metric_values)

    # Calculate a 10% margin
    y_margin = (max_val - min_val) * 0.1 if (max_val - min_val) != 0 else 0.05

    # Set the dynamic y-axis limits with the margin
    ax.set_ylim(min_val - y_margin, max_val + y_margin)



    # Add grid
    ax.grid(True, linestyle='--', axis='y', alpha=0.7)

# Add a single common legend
handles = [plt.Line2D([0], [0], color=color, lw=4) for color in colors.values()]
fig.legend(handles, dataset_labels, loc='upper center', ncol=len(colors))

# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust top spacing for the legend
plt.savefig('Results/plot_bce.png', dpi=400)
plt.show()

