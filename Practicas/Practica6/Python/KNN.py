from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def KNN(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        # Initialize K-Nearest Neighbors classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=3)  # You can experiment with different values of 'n_neighbors'
        # Train the model
    knn_classifier.fit(X_train, y_train)
        # Make predictions on the test set
    y_pred = knn_classifier.predict(X_test)

        # Evaluate the model
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
       

    # Display confusion matrix

    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()
    print(f'Accuracy: {accuracy * 100:.2f}%')
    return y_pred