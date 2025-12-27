import pickle as pkl
from time import time
import libsvm as svm

train_dataset_path = "/content/output/encoded_train.bin"
test_dataset_path = "/content/output/encoded_test.bin"
N_CLASSES = 10

class Dataset:
    def __init__(self, data, labels, n, width, height, depth):
        self.data = data          # The list of images (flattened, shape=(32, 32, 3))
        self.labels = labels      # The list of the labels (int)
        self.n = n                # The number of images in the list
        self.width = width        # The width of the image
        self.height = height      # The height of the image
        self.depth = depth        # The depth of the image

    def load_data(self, path):
        with open(path, 'rb') as f:
            dataset = pkl.load(f)
        self.data = dataset['data']
        self.labels = dataset['labels']
        self.n = dataset['n']
        self.width = dataset['width']
        self.height = dataset['height']
        self.depth = dataset['depth']

# Train SVM (LibSVM)               
# ✓ Input: train_features + labels                 
# ✓ Kernel: RBF (Radial Basis Function)           
# ✓ Hyperparameters: C=10, gamma=auto             
# ✓ Output: trained SVM model  
                
# Evaluate                                 
# ✓ Predict on test_features using SVM             
# ✓ Calculate accuracy, confusion matrix           
# ✓ Expected accuracy: 60-65%                      
# ✓ Compare with baseline methods

KERNEL = 'rbf'
C = 10.0
GAMMA = 'auto'

class SVMModel:
    def __init__(self):
        self.model = None

    def train(self, features, labels):
        problem = svm.svm_problem(labels, features)
        parameters = svm.svm_parameter(f'-s 0 -t 2 -c {C} -g {GAMMA}')
        self.model = svm.svm_train(problem, parameters)

    def predict(self, features):
        predictions = []
        for feature in features:
            label, _, _ = svm.svm_predict([0], [feature], self.model)
            predictions.append(label[0])
        return predictions
    
    def evaluate(self, predictions, true_labels):
        correct = sum(p == t for p, t in zip(predictions, true_labels))
        accuracy = correct / len(true_labels)
        
        # Confusion matrix
        num_classes = len(set(true_labels))
        confusion_matrix = [[0] * num_classes for _ in range(num_classes)]
        for p, t in zip(predictions, true_labels):
            confusion_matrix[int(t)][int(p)] += 1
        
        return accuracy, confusion_matrix

    def save_model(self, path):
        with open(path, 'wb') as f:
            pkl.dump(self.model, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            self.model = pkl.load(f)

    def save_evaluation_results(self, accuracy, confusion_matrix, path):
        with open(path, 'w') as f:
            f.write(f'Accuracy: {accuracy:.4f}\n')
            f.write('Confusion Matrix:\n')
            for row in confusion_matrix:
                f.write(' '.join(f'{val:5d}' for val in row) + '\n')
    
def print_confusion_matrix(confusion_matrix):
    print("Confusion Matrix:")
    for row in confusion_matrix:
        print(" ".join(f"{val:5d}" for val in row))

def main():
    trainset = Dataset(None, None, 0, 0, 0, 0)
    trainset.load_data(train_dataset_path)

    testset = Dataset(None, None, 0, 0, 0, 0)
    testset.load_data(test_dataset_path)

    svm_model = SVMModel()
    print("Training SVM model...")
    traintime_start = time.time()
    svm_model.train(trainset.data, trainset.labels)
    traintime_end = time.time()
    print(f"Training time: {traintime_end - traintime_start:.2f} seconds")
    print("Training completed.")

    print("Evaluating SVM model...")
    predtime_start = time.time()
    predictions = svm_model.predict(testset.data)
    predtime_end = time.time()
    print(f"Prediction time: {predtime_end - predtime_start:.2f} seconds")
    accuracy, confusion_matrix = svm_model.evaluate(predictions, testset.labels)
    print(f"Accuracy: {accuracy:.4f}") 
    print_confusion_matrix(confusion_matrix)

    # Save model and evaluation results
    svm_model.save_model("svm_model.pkl")
    svm_model.save_evaluation_results(accuracy, confusion_matrix, "svm_evaluation.txt")

if __name__ == "__main__":
    main()