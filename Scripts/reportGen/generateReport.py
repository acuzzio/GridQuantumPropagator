import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader
import webbrowser
import os
import quantumpropagator as qp
import io
import base64
import matplotlib.pyplot as plt

def fig_to_html(fig):
    '''
    This function takes a matplotlib Figure object and returns the html png string
    '''
    img = io.BytesIO()
    fig.savefig(img, format='png',
                bbox_inches='tight'
                )
    img.seek(0)
    encoded = base64.b64encode(img.getvalue())
    string_html = '<img src="data:image/png;base64, {}">'
    figure_code = string_html.format(encoded.decode('utf-8'))
    return figure_code

def info_coord(dictio):
    '''
    Creates a string for the coordinates
    '''

    phis_ext = dictio['phis']
    gams_ext = dictio['gams']
    thes_ext = dictio['thes']

    #phiV_ext, gamV_ext, theV_ext = dictio

    # take step
    dphi = phis_ext[0] - phis_ext[1]
    dgam = gams_ext[0] - gams_ext[1]
    dthe = thes_ext[0] - thes_ext[1]

    # take range
    range_phi = phis_ext[-1] - phis_ext[0]
    range_gam = gams_ext[-1] - gams_ext[0]
    range_the = thes_ext[-1] - thes_ext[0]

    columns_df =  [ 'sx_extr', 'dx_extr', 'dq', 'range' ]
    little_table = {
                   'phi' : [phis_ext[-1],phis_ext[0],dphi,range_phi],
                   'gam' : [gams_ext[-1],gams_ext[0],dgam,range_gam],
                   'the' : [thes_ext[-1],thes_ext[0],dthe,range_the]
                   }
    coordinates_df = pd.DataFrame.from_dict(little_table, orient='index')
    coordinates_df.columns = columns_df
    return (coordinates_df.to_html())

def info_pulse(dictio):
    '''
    Creates a string with a table for the pulse
    '''
    columns_df = ['E','omega','sigma','phi','t0']
    little_table = {
            'X' : dictio['pulseX'],
            'Y' : dictio['pulseY'],
            'Z' : dictio['pulseZ']}
    pulse_df = pd.DataFrame.from_dict(little_table, orient='index')
    pulse_df.columns = columns_df
    return (pulse_df.to_html())

def create_string_input(allout):
    '''
    This function transform the all input h5 file into a string of information for the report
    '''
    dictio = qp.readWholeH5toDict(allout)
    # 'theL', 'dphi', 'nacCube', 'nstates', 'pulseZ', 'pulseX', 'kind', 'natoms', 'kinCube', 'outFol', 'dipCube', 'phis', 'fullTime', 'dgam', 'dthe', 'h', 'gams', 'potCube', 'phiL', 'thes', 'gamL', 'pulseY'
    coord_string = 'Coordinates:' + info_coord(dictio)
    pulse_string = 'Pulse specs in AU:' + info_pulse(dictio)

    fullString = coord_string + pulse_string
    return(fullString)

def main():
    '''
    Transform a dynamics folder into a html report
    '''
    nstates = 8

    # style part
    style = 'home/alessio/y-RepoQuantum/Scripts/reportGen/style.css'
    with open('style.css','r') as f:
        style_string = f.read()


    root = '/home/alessio/m-dynamicshere/results'
    #project = 'w-from6withouspulse_0000'
    #project = 'x-from5withouspulse_0000'
    #project = 'y-from4withouspulse_0000'
    project = 'z-from3withouspulse_0000'
    folder = os.path.join(root, project)
    allout = os.path.join(folder,'allInput.h5')
    outfn = os.path.join(folder,'output')
    outfnP = os.path.join(folder,'outputPopul')

    info_string = create_string_input(allout)
    # dictio = qp.readWholeH5toDict(allout)

    env = Environment(loader=FileSystemLoader('.'))
    template = env.get_template("report.html.j2")


    data = pd.read_csv(outfn, delim_whitespace=True, header=None);
    dataP = pd.read_csv(outfnP, delim_whitespace=True, header=None);
    data.columns = ['count','steps','fs','Norm Deviation','Kinetic','Potential','Total','Total deviation','Xpulse','Ypulse','Zpulse']
    result = pd.concat([data, dataP], axis=1);

    df2 = pd.DataFrame(result)

    # title
    title_Repo = 'Report: {}'.format(project)

    # first graph
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.set_ylabel('Population')
    ax2.set_ylabel('Pulse')
    popul = np.arange(nstates)+1
    result.plot(title = 'Population and Pulse', ax = ax1,  x=['fs'], y=popul, linewidth=0.8)
    result.plot(title = 'Population and Pulse', ax = ax2,  x=['fs'], y=['Xpulse','Ypulse','Zpulse'], linewidth=0.5,ls='--', legend=False);

    popul_figure = fig_to_html(fig)

    # second figure
    fig2 = plt.figure(figsize=(15,4))
    ax1 = fig2.add_subplot(111)
    ax1.set_ylabel('Units')
    data.plot(title = 'Norm Deviation', ax=ax1, x='fs', y = 'Norm Deviation')
    norm_figure = fig_to_html(fig2)


    # third figure
    fig3 = plt.figure(figsize=(15,4))
    ax1 = fig3.add_subplot(111)
    ax1.set_ylabel('Hartree')
    data['Kinetic_Moved'] = data['Kinetic'] + data['Potential'][0]
    data.plot(title = 'Comparison Potential Total Kinetic', ax=ax1, x=['fs'] ,y=['Kinetic_Moved','Potential','Total'], figsize=(15,5))

    kin_tot_figure = fig_to_html(fig3)

    # setting the html
    template_vars = {"title" : title_Repo,
                     "table_output": df2.to_html(),
                     "info_string": info_string,
                     "popul_figure": popul_figure,
                     "kin_tot_figure": kin_tot_figure,
                     "style_string": style_string,
                     "norm_figure": norm_figure}

    html_out = template.render(template_vars)

    filename = 'Report_{}.html'.format(project)

    with open(filename, 'w') as f:
        f.write(html_out)

    print('\nFile {} written.\n'.format(filename))

    # open the browser or not
    webbrowser.open(filename)



if __name__ == "__main__":
    main()
