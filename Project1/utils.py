import matplotlib.pyplot as plt
import numpy as np

# Data from the image
labels = ['FC', 'All layers', 'No layers']
train_set = [92.00, 97.68, 54.25]
test_set = [88.78, 85.45, 49.68]

x = np.arange(len(labels))  # label locations
width = 0.35  # width of the bars

# Create the plot with larger text and a more condensed layout
fig, ax = plt.subplots(figsize=(8, 5))

# Bar chart with condensed layout
rects1 = ax.bar(x - width/2, train_set, width, label='Train Set', color='royalblue')
rects2 = ax.bar(x + width/2, test_set, width, label='Test Set', color='orange')

# Adding text and adjusting layout
ax.set_xlabel('Layers Unfrozen for Transfer Learning', fontsize=14)
ax.set_ylabel('Accuracy (%)', fontsize=14)
ax.set_title('Training and Test Set Performance', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=12)
ax.legend(fontsize=12)

# Attach accuracy labels to bars
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', fontsize=12)

add_labels(rects1)
add_labels(rects2)

# Condense the layout
fig.tight_layout()

# Display the updated plot
plt.show()