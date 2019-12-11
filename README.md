# Devanagari-Handwritten-Character-Dataset
The dataset can be downloaded from: https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset#

## Accuracies of the models created:

| Model | Accuracy |
| ----- | -------- |
| Logistic Regression | 65.9% |
| SVM with degree 2 Polynomial Kernel | 82.52% |
| KNN | 83% |
| Random Forest | 83.1% |
| Covolutional Autoencoder (CAE) | 95.09% |
| CAE+SVM | 65.9% |
| CNN | 98.83 |
| CNN+SVM | 99.06% |

For more details, see the Report.

### Contributors: 
1. [anandthegreat](https://github.com/anandthegreat)
2. [atul2938](https://github.com/atul2938)
3. [singh4akash](https://github.com/singh4akash)

###### To convert the raw [dataset](https://archive.ics.uci.edu/ml/datasets/Devanagari+Handwritten+Character+Dataset#) to csv format, use the following code:
```python
root_dir = os.getcwd()
img_dir = [os.path.join(root_dir, 'train'),os.path.join(root_dir, 'test')]

pixels = np.array(['pixel_{:03d}'.format(x) for x in range(784)])

for i in range(0,2):
    for char_name in sorted(os.listdir(img_dir[i])):
        char_dir = os.path.join(img_dir[i], char_name)
        img_df = pd.DataFrame(columns=pixels)

        for img_file in sorted(os.listdir(char_dir)):
            image = imread(os.path.join(char_dir, img_file))
            image = image[2:30,2:30]
            image = pd.Series(image.flatten(),index = pixels)
            img_df = img_df.append(image.T, ignore_index=True)

        img_df = img_df.astype(np.uint8)
        img_df['character'] = char_name
        
        if(i==0):
            img_df.to_csv('data.csv', index=False, mode='a', header=flag)
        else:
            img_df.to_csv('data_test.csv', index=False, mode='a', header=flag)
```
or simply run the SVM code.
