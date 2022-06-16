import os
import random
from pprint import pprint

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pandas as pd

from crossover import crossover_with_class
from preprocessed import get_all_files_text
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import streamlit as st


def get_all_preprocessed_files(preprocessed_folder):
    files = []
    for obj in os.listdir(preprocessed_folder):
        path = preprocessed_folder + obj + '/'
        for file in os.listdir(path):
            files.append(f'{path}{file}')
    return files


def get_all_files_text(preprocessed_folder):
    # возвращает название класса и текст документа
    paths = get_all_preprocessed_files(preprocessed_folder)
    for path in paths:
        with open(path, 'r') as file:
            file_class = path.split('/')[-2]
            yield file_class, file.read()


def get_files(preprocessed_folder, class_min_elem_count=0):
    files = get_all_files_text(preprocessed_folder)
    files = list(files)
    print(len(files))
    r = {}
    for file in files:
        l = r.get(file[0], [])
        r[file[0]] = l
        l.append(file[1])

    # print(r)
    labels_name = [e for e in r.keys()]
    print(labels_name)
    r = [e for e in r.items() if len(e[1]) > class_min_elem_count]
    print(len(r))
    files = [[e[0], i] for e in r for i in e[1]]
    print(len(files))

    tf_texts = [e[1] for e in files]
    score_list = [e[0] for e in files]
    return tf_texts, score_list


def create_vectors(tf_texts, score_list, crossover=None, crossover_max_count=None):
    X_train, X_test, y_train, y_test = train_test_split(tf_texts, score_list,
                                                        test_size=0.2, random_state=42,
                                                        stratify=score_list)
    print(sorted(y_test))
    if crossover:
        max_count = crossover_max_count if crossover_max_count else 30
        X_train, y_train = crossover_with_class(X_train, y_train, max_count=max_count)
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1),
                                 min_df=0, stop_words='english', sublinear_tf=True)
    # vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 1),
    #                              min_df=0, stop_words='english')
    response = vectorizer.fit_transform(X_train)
    test_vectorize = vectorizer.transform(X_test)
    return vectorizer, response, test_vectorize, y_train, y_test


def create_and_fit_model(text_vectors, classes, c=1):
    clf = LinearSVC(random_state=42, C=c)
    clf.fit(text_vectors, classes)
    return clf


def plot_confussing_matrix(y_test, y_pred, name_added=None):
    titles_options = [("Confusion_matrix_without_normalization", None),
                      ("Normalized_confusion_matrix", 'true')]
    # figures = []
    plt.rcParams['font.size'] = '2'
    for title, normalize in titles_options:
        cm = confusion_matrix(y_test, y_pred,
                              normalize=normalize)

        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues, include_values=True)
        disp.ax_.set_title(title)
        plt.tick_params(axis='both', which='major')
        plt.savefig('{}{}'.format('', f'{title}{name_added}'), dpi=1000)
        # fig = plt.gcf()
        # figures.append(fig)
    # return figures


def get_f1_presicion_recall(y_pred, y_test):
    unique_classes = sorted(list(set(y_test)))
    res = precision_recall_fscore_support(y_test, y_pred, average=None, labels=unique_classes)
    results = {}
    for i in range(len(unique_classes)):
        results[unique_classes[i]] = [elem[i] for elem in res]
    res = precision_recall_fscore_support(y_test, y_pred, average='macro')
    results['all'] = res
    return results


def test_model(model, x_test, y_test, name_added):
    y_pred = model.predict(x_test)
    results = get_f1_presicion_recall(y_pred, y_test)
    pprint(results)
    figures = plot_confussing_matrix(y_test, y_pred, name_added)
    accuracy = sum([e[0] == e[1] for e in zip(y_pred, y_test)]) / len(y_test)
    print('Accuracy: %f' % (accuracy * 100))
    return results, accuracy


def main():
    st.sidebar.subheader('Параметры модели')
    x = st.sidebar.slider('C', value=1.0, min_value=0.0, max_value=2.0)  # 👈 this is a widget
    x2 = st.sidebar.slider('Минимальное число элементов в классе', value=0, min_value=0, max_value=100)
    x3 = bool(st.sidebar.checkbox('Кроссовер'))
    print('x3', x3)
    x4 = st.sidebar.slider('Расширить количество элементов в классе до', value=30, min_value=0, max_value=100)

    eng_files_preprocessed_folder = 'preprocessed_files/'
    ru_test_files_preprocessed_folder = 'collections_dml/collection_for_testing_msc_preprocessed/'
    files, classes = get_files(eng_files_preprocessed_folder, class_min_elem_count=x2)
    vectorizer, x_train, x_test, y_train, y_test = create_vectors(files, classes, crossover=x3, crossover_max_count=x4)
    model = create_and_fit_model(x_train, y_train, c=x)
    data1, accuracy1 = test_model(model, x_test, y_test, '1')

    texts, classes = get_files(ru_test_files_preprocessed_folder)
    texts_vectors = vectorizer.transform(texts)
    data2, accuracy2 = test_model(model, texts_vectors, classes, '2')

    st.subheader('Результаты тестирования на англоязычных данных\n')
    st.text('Таблица метрик по каждому классу')
    d1 = pd.DataFrame(data1)
    d1 = d1.T
    d1.columns = ['precision', 'recall', 'f1', 'count']
    d1
    st.text('Таблица метрик по всем классам')
    dd1 = pd.DataFrame({'all': data1['all']}).T
    dd1.columns = ['precision', 'recall', 'f1', 'count']
    dd1
    st.text(f'Accuracy: {accuracy1}')
    st.text('Нормализованная матрица сопряженности')
    img = mpimg.imread('Normalized_confusion_matrix1.png')
    st.image(img)
    st.text('Количественная матрица сопряженности')
    img = mpimg.imread('Confusion_matrix_without_normalization1.png')
    st.image(img)

    st.subheader('Результаты тестирования на русскоязычных данных\n')
    st.text('Таблица метрик по каждому классу')
    d2 = pd.DataFrame(data2)
    d2 = d2.T
    d2
    st.text('Таблица метрик по всем классам')
    d2.columns = ['precision', 'recall', 'f1', 'count']
    dd2 = pd.DataFrame({'all': data2['all']}).T
    dd2.columns = ['precision', 'recall', 'f1', 'count']
    dd2
    st.text(f'Accuracy: {accuracy2}')
    st.text('Нормализованная матрица сопряженности')
    img = mpimg.imread('Normalized_confusion_matrix2.png')
    st.image(img)
    st.text('Количественная матрица сопряженности')
    img = mpimg.imread('Confusion_matrix_without_normalization2.png')
    st.image(img)


if __name__ == '__main__':
    main()
