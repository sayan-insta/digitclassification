# main.py

from utils import (load_and_visualize_data, prepare_data, train_classifier,
                   predict_and_visualize, evaluate_classifier,
                   plot_confusion_matrix, rebuild_report_from_confusion_matrix)

def main():
    digits = load_and_visualize_data()
    X_train, X_test, y_train, y_test = prepare_data(digits)
    clf = train_classifier(X_train, y_train)
    predicted = predict_and_visualize(clf, X_test)
    evaluate_classifier(clf, y_test, predicted)
    cm = plot_confusion_matrix(y_test, predicted)
    rebuild_report_from_confusion_matrix(cm)

if __name__ == "__main__":
    main()
