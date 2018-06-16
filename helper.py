import matplotlib.pyplot as plt
import numpy as np

def test_classification(model, test_data, test_target, index):
    img = test_data[index].reshape(8,8)
    pred = model.predict_proba(test_data[index:index+1]).squeeze()
    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    
    ax1.imshow(img, cmap='gray')
    ax1.set_title('Actual: ' + str(test_target[index]))
    ax1.axis('off')
    
    ax2.barh(np.arange(10), pred)
    ax2.set_aspect(0.1)
    ax2.set_yticks(np.arange(10))
    ax2.set_yticklabels(np.arange(10))
    ax2.set_title('Model Prediction')
    ax2.set_xlim(0, 1.1)
    ax2.invert_yaxis()

    plt.tight_layout()