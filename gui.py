import PySimpleGUI as sg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path

def read_table():
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text('Datasets Directory', size=(16, 1)), sg.InputText(),
               sg.FolderBrowse()],
              [sg.Submit(), sg.Cancel(),
               sg.Checkbox('CSV', size=(15,1), key='csv', default=False),
               sg.Checkbox('PLK', size=(15,1), key='plk', default=False)]]

    window1 = sg.Window('Input file', layout)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return

    path = values['Browse']

    if path == '':
        return

    basepath = Path(path)
    files_in_basepath = basepath.iterdir()

    if path is not None:
        try:
            datasets = {}
            for item in files_in_basepath:
                if item.is_file():
                    if values['plk']:
                        df = pd.read_pickle(path + '/' + item.name)
                        name_base = item.name.replace('.plk', '')
                    elif values['csv']:
                        df = pd.read_csv(path + '/' + item.name)
                        name_base = item.name.replace('.cv', '')

                    datasets[name_base] = df

            window1.close()
            return datasets
        except:
            sg.popup_error(
                'Error reading file')
            window1.close()
            return


def show_table(data, header_list, name):
    layout = [
        [sg.Table(values=data,
                  headings=header_list,
                  font='Helvetica',
                  pad=(25, 25),
                  display_row_numbers=False,
                  auto_size_columns=True,
                  num_rows=min(25, len(data)))]
    ]

    window = sg.Window(name, layout, grab_anywhere=False)
    event, values = window.read()
    window.close()

def results_frame(dic):

    list_dataset = []
    for dataset in dic.keys():
        list_dataset.append(sg.Checkbox(dataset, size=(10, 1), key=dataset, default=False))
    metrics = ['precision', 'recall', 'F1', 'auc_roc', 'accuracy']
    list_metric = []
    for metric in metrics:
        list_metric.append(sg.Checkbox(metric, size=(10, 1), key=metric, default=False))

    layout = [list_dataset,
              list_metric,
              [sg.Button('Table', size=(10, 1), enable_events=True, key='table', font='Helvetica 16'),
               sg.Button('Bar Graphic', size=(10, 1), enable_events=True, key='graphic', font='Helvetica 16')]]

    window = sg.Window('Select results', layout, size=(600, 150))

    while True:
        event, values = window.read()

        if event in (sg.WIN_CLOSED, 'Exit'):
            break
        if event == 'table':
            try:
                for dataset in dic.keys():
                    if values[dataset]:
                        for metric in metrics:
                            if values[metric]:
                                df = return_df(dataset,metric)
                                header_list = list(df.columns)
                                data = df[0:].values.tolist()
                                show_table(data, header_list, dataset + '-' + metric )
            except:
                pass
        if event == 'graphic':
            try:
                for dataset in dic.keys():
                    if values[dataset]:
                        for metric in metrics:
                            if values[metric]:
                                df = return_df(dataset, metric)
                                show_graphic(df, dataset + '-' + metric)
            except:
                pass

def results(dic, path, item):

    name = item.name.split('_')

    tam = len(name)

    dataset = name[0]

    percent = name[tam - 2]

    if name[1] == 'BoW':
        if name[2] == 'term-frequency-IDF':
            preprocessing = 'Bow-TFIDF'
        elif name[2] == 'term-frequency':
            preprocessing = 'Bow-TF'
        elif name[2] == 'binary':
            preprocessing = 'Bow-Binary'
    elif name[1] == 'DensityInformation':
        preprocessing = 'Density'
    else:
        preprocessing = name[1]

    df = pd.read_csv(path + '/' + item.name, sep=';')

    best_f1 = np.max(df['f1-score'])

    best_pre = df[df['f1-score'] == best_f1]['precision'].iloc[0]

    brest_rev = df[df['f1-score'] == best_f1]['recall'].iloc[0]

    best_aucroc = np.max(df['auc_roc'])

    best_accuracy = np.max(df['accuracy'])

    if dataset not in dic:
        dic[dataset] = {}
        dic[dataset][preprocessing] = pd.DataFrame(
            columns=['percent', 'precision', 'recall', 'F1', 'auc_roc', 'accuracy'])
    elif preprocessing not in dic[dataset]:
        dic[dataset][preprocessing] = pd.DataFrame(
            columns=['percent', 'precision', 'recall', 'F1', 'auc_roc', 'accuracy'])

    df_bests = dic[dataset][preprocessing]

    df_bests = df_bests.append({'percent': percent.replace('.csv', '_%'),
                                'precision': best_pre,
                                'recall': brest_rev,
                                'F1': best_f1,
                                'auc_roc': best_aucroc,
                                'accuracy': best_accuracy},
                               ignore_index=True)

    dic[dataset][preprocessing] = df_bests

def return_df(dataset, metric):
  df = pd.DataFrame(columns=['percent'] + list(dic[dataset].keys()))
  df['percent'] = dic[dataset]['AE']['percent']

  for prepro in dic[dataset].keys():
    df[prepro] = dic[dataset][prepro][metric]

  return df

def read_results():
    sg.set_options(auto_size_buttons=True)
    layout = [[sg.Text('Results Directory', size=(16, 1)), sg.InputText(),
               sg.FolderBrowse()],
              [sg.Submit(), sg.Cancel(),
               sg.Checkbox('Percents', size=(15,1), key='percent', default=False)]]

    window1 = sg.Window('Input file', layout)
    try:
        event, values = window1.read()
        window1.close()
    except:
        window1.close()
        return

    path = values['Browse']

    if path == '':
        return

    basepath = Path(path)
    files_in_basepath = basepath.iterdir()

    if path is not None:
        try:
            dic = {}
            for item in files_in_basepath:
                if item.is_file():
                    results(dic, path, item)

            window1.close()
            return dic
        except:
            sg.popup_error(
                'Error reading file')
            window1.close()
            return

def draw_figure(canvas, figure, loc=(0, 0)):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def show_graphic(df, name):

    N = len(list(df.keys())[1:])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.2  # the width of the bars

    fig = plt.figure(figsize=(15, 7))
    ax = fig.add_subplot(111)

    yvals = list(df[df['percent'] == '0.25'][list(df.keys())[1:]].iloc[0])

    rects1 = ax.bar(ind, yvals, width, color=(0.1, 0.5, 0.5, 0.4))

    zvals = list(df[df['percent'] == '0.5'][list(df.keys())[1:]].iloc[0])

    rects2 = ax.bar(ind + width, zvals, width, color=(0.1, 0.5, 0.5, 0.6))

    kvals = list(df[df['percent'] == '0.75'][list(df.keys())[1:]].iloc[0])

    rects3 = ax.bar(ind + width * 2, kvals, width, color=(0.1, 0.5, 0.6, 0.8))

    gvals = list(df[df['percent'] == '1'][list(df.keys())[1:]].iloc[0])

    rects4 = ax.bar(ind + width * 3, gvals, width, color=(0.1, 0.5, 0.8, 1))

    ax.set_ylabel('F1-Score')
    ax.set_xlabel('Preprocessing Technique')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(list(df.keys())[1:])
    ax.legend((rects1[0], rects2[0], rects3[0], rects4[0]), ('0.25', '0.5', '0.75', '1.0'))

    layout = [[sg.Canvas(size=(15, 7), key='-CANVAS-')]]

    window = sg.Window(name, layout, layout, force_toplevel=True, finalize=True)

    fig_photo = draw_figure(window['-CANVAS-'].TKCanvas, fig)

    event, values = window.read()
    window.close()


layout = [
    [sg.Button('Datasets', size=(15, 1), enable_events=True, key='Datasets', font='Helvetica 16'),
     sg.Button('Preprocessing', size=(15, 1), enable_events=True, key='preprocessing', font='Helvetica 16'),
     sg.Button('Classification', size=(15, 1), enable_events=True, key='classification', font='Helvetica 16'),
     sg.Button('Results', size=(15, 1), enable_events=True, key='results', font='Helvetica 16')],
     [sg.Graph(canvas_size=(800, 500), graph_bottom_left=(0, 0), graph_top_right=(500, 800), key="graph")]
]

window = sg.Window('One-Class Text Categorization', layout, size=(900, 450))
read_successful = False
window.Finalize()
graph = window.Element("graph")
graph.DrawImage(filename="./images/Pipeline-quali.png", location=(50, 700))

results_dic = False
# Event loop
while True:
    event, values = window.read()

    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if event == 'Datasets':
        try:
            datasets = read_table()
        except:
            pass
    if event == 'preprocessing':
        print('preprocessing here TO DO')
    if event == 'classification':
        print('classification here TO DO')
    if event == 'results':
        if not results_dic:
            dic = read_results()
        results_frame(dic)
