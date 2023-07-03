# Ramiro Isa-Jara, ramiro.isaj@gmail.com
# GUI Interface to detect spheres of cells in images using
# Normalize image, Adaptive threshold and detection of contours

import time
import numpy as np
import pandas as pd
import PySimpleGUI as sg
import Sphere_def as Sp
from datetime import datetime

# -------------------------------
# Adjust size screen
# -------------------------------
Screen_size = 10
# -------------------------------
sg.theme('LightGrey1')
l_res = ['100x', '90x', '60x', '40x', '20x', '10x', '4x']
l_conv = [15.50, 13.95, 9.30, 6.21, 3.10, 1.55, 1.09]

layout1 = [[sg.Radio('Windows', "RADIO1", enable_events=True, default=True, key='_SYS_')],
           [sg.Radio('Linux', "RADIO1", enable_events=True, key='_LIN_')], [sg.Text('')]]

layout2 = [[sg.Checkbox('*.jpg', default=True, key="_IN1_")], [sg.Checkbox('*.png', default=False, key="_IN2_")],
           [sg.Checkbox('*.tiff', default=False, key="_IN3_")]]

layout3 = [[sg.Text('Filter parameter:', size=(12, 1)), sg.InputText('41', key='_IDI_', size=(7, 1))],
           [sg.Text('Image resolution:', size=(12, 1)), sg.Combo(l_res, size=(5, 1), default_value='100x', key='_RES_'),
            sg.Text('', size=(1, 1))],
           [sg.Checkbox('Save Processed Images', default=True, enable_events=True, key='_SIM_')]]

layout4 = [[sg.Text('Name Outfile: ', size=(12, 1)), sg.InputText('Experiment1_', size=(28, 1), key='_NAM_')],
           [sg.Text('Path Images: ', size=(12, 1)), sg.InputText(size=(35, 1), key='_ORI_'), sg.FolderBrowse()],
           [sg.Text('Path Results: ', size=(12, 1)), sg.InputText(size=(35, 1), key='_DES_'), sg.FolderBrowse()]]

layout5 = [[sg.T("", size=(15, 1)), sg.Text('         NO PROCESS', size=(33, 1), key='_MES_', text_color='DarkRed'),
            sg.Text('', size=(1, 1))]]

layout6 = [[sg.T("", size=(14, 1)), sg.Text('Current time: ', size=(11, 1), font=('Arial', 9, 'bold')),
            sg.Text('', size=(10, 1), key='_TAC_')],
           [sg.T("", size=(2, 1)),
            sg.Text('Start time: ', size=(8, 1), font=('Arial', 9, 'bold')),
            sg.Text('-- : -- : --', size=(10, 1), key='_TIN_', text_color='blue'), sg.T("", size=(6, 1)),
            sg.Text('Finish time: ', size=(9, 1), font=('Arial', 9, 'bold')),
            sg.Text('-- : -- : --', size=(10, 1), key='_TFI_', text_color='red')],
           [sg.Text('Name of image: ', size=(12, 1)), sg.InputText('', key='_NIM_', size=(39, 1))],
           [sg.Text('Current image: ', size=(12, 1)), sg.InputText('', key='_CUR_', size=(10, 1)),
            sg.Text('', size=(2, 1)),
           sg.Text('Total images: ', size=(11, 1)), sg.InputText('', key='_CIM_', size=(8, 1))],
           [sg.Text('Nr. of Spheres: ', size=(12, 1)), sg.InputText('', key='_SPH_', size=(10, 1)),
            sg.Text('', size=(2, 1)),
            sg.Text('Time used: ', size=(11, 1)), sg.InputText('', key='_TUS_', size=(8, 1)),
            sg.Text('...', size=(3, 1), key='_IDE_')],
           [sg.Text('Image area: ', size=(12, 1)), sg.InputText('', key='_TAR_', size=(10, 1)),
            sg.Text('um2', size=(4, 1)),],
           [sg.Text('Detected area: ', size=(12, 1)), sg.InputText('', key='_DAR_', size=(10, 1)),
            sg.Text('um2', size=(3, 1)),
            sg.Text('Percentage: ', size=(10, 1)), sg.InputText('', key='_PAR_', size=(8, 1)),
            sg.Text('%', size=(3, 1)),]
           ]


v_image = [sg.Image(filename='', key="_IMA_")]
# columns
col_1 = [[sg.Frame('', [v_image])]]
col_2 = [[sg.Frame('Operative System: ', layout1, title_color='Blue'),
          sg.Frame('Type image: ', layout2, title_color='Blue'), sg.Frame('Settings: ', layout3, title_color='Blue')],
         [sg.Frame('Directories: ', layout4, title_color='Blue')],
         [sg.Text(" ", size=(13, 1)), sg.Button('Start', size=(8, 1)),
          sg.Button('Pause', size=(8, 1)), sg.Button('Finish', size=(8, 1))],
         [sg.Frame('', layout5)], [sg.Frame('', layout6)]]

layout = [[sg.Column(col_1), sg.Column(col_2)]]

# Create the Window
window = sg.Window('Spheres Interface', layout, font="Helvetica "+str(Screen_size), finalize=True)
# ----------------------------------------------------------------------------------
time_, id_image, time_h, time_l, fluid_h, fluid_l, port_name, bauds_, c_port, i = 0, 0, 0, 0, 0, 0, 0, 0, -1, 0
start_, save_, pump_, control, finish_, pause_ = False, True, False, True, False, False
video, name, image, ini_time, ini_time_, path_des, type_i, path_ori = None, None, None, None, None, None, None, None
saveIm, pumpC, filenames, id_sys, h_filter, name_file, conv_value = None, None, [], 0, 41, None, 0
m1, n1 = 450, 400
results = pd.DataFrame(columns=['Image', 'Sphere', 'Detected Area (um2)', 'Percentage Area',
                                'Image Area (um2)', 'Time (sec)'])
# ----------------------------------------------------------------------------------
img = np.ones((m1, n1, 1), np.uint8)*255
sphere = Sp.SphereImages(window, m1, n1)
window['_IMA_'].update(data=sphere.bytes_(img))
# -----------------------------------------------------------------------------------

# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read(timeout=50)
    window.Refresh()
    now = datetime.now()
    now_time = now.strftime("%H : %M : %S")
    window['_TAC_'].update(now_time)

    if event == sg.WIN_CLOSED:
        break

    if event == 'Finish' or finish_:
        print('FINISH PROCESS')
        window['_MES_'].update('    PROCESS IS FINISHED  ')
        now_time = now.strftime("%H : %M : %S")
        window['_TFI_'].update(now_time)
        if start_:
            sphere.save_csv_file(results, path_des, name_file)
        start_, finish_ = False, False

    if event == 'Pause':
        if not pause_:
            start_ = False
            pause_ = True
        else:
            start_ = True
            pause_ = False

    if event == '_SIM_':
        save_ = values['_SIM_']

    if event == 'Start':
        print('START ANALYSIS')
        now_time = now.strftime("%H : %M : %S")
        window['_TIN_'].update(now_time)
        if values['_SYS_']:
            path_ori = sphere.update_dir(values['_ORI_']) + "\\"
            path_ori = r'{}'.format(path_ori)
            path_des = sphere.update_dir(values['_DES_']) + "\\"
            path_des = r'{}'.format(path_des)
            id_sys = 0
        else:
            path_ori = values['_ORI_'] + '/'
            path_des = values['_DES_'] + '/'
            id_sys = 1
        # -------------------------------------------------------------------
        window['_IDE_'].update('sec')
        # -------------------------------------------------------------------
        if values['_IN2_']:
            type_i = ".png"
        elif values['_IN3_']:
            type_i = ".tiff"
        else:
            type_i = ".jpg"
        # ------------------------------------------------------------------
        if len(path_ori) > 1 and len(path_des) > 1 and start_ is False:
            start_ = True
            ini_time = datetime.now()
            name_file = values['_NAM_']
            index = l_res.index(values['_RES_'])
            conv_value = l_conv[index]
            h_filter = int(values['_IDI_'])
        elif len(path_ori) > 1 and len(path_des) > 1 and start_:
            sg.Popup('Warning', ['Analysis is running...'])
        else:
            sg.Popup('Error', ['Information or process is incorrect'])

    if start_:
        filenames, image_, filename, total_i = sphere.load_image_i(path_ori, i, type_i, filenames, id_sys)
        window['_CIM_'].update(total_i)

        if len(image_) == 0 and total_i == 0:
            finish_ = True
            sg.Popup('Error', ['No images in directory. '])
        elif i == total_i:
            finish_ = True
            continue
        else:
            window['_CUR_'].update(i+1)
            window['_CIM_'].update(total_i)
            window['_NIM_'].update(filename)
            window['_MES_'].update(' ... IMAGE PROCESSING .... ')
            print('|-------------------------------------------------------|')
            print('Processing image: ... ' + str(i+1) + ' of ' + str(total_i))
            image_out, results, a_total, a_detected, percentage, n_spheres, time_p = sphere.sphere_main(image_, i,
                                                                                                        h_filter,
                                                                                                        results,
                                                                                                        conv_value)
            if save_:
                sphere.save_image_out(image_out, path_des, filename)

            window['_TAR_'].update(a_total)
            window['_DAR_'].update(a_detected)
            window['_PAR_'].update(percentage)
            window['_SPH_'].update(n_spheres)
            window['_TUS_'].update(time_p)
            window['_IMA_'].update(data=sphere.bytes_(image_out))
            i += 1
            time.sleep(0.10)

    if pause_:
        window['_MES_'].update('  Pulse PAUSE to resume ...')

print('CLOSE WINDOW')
window.close()
