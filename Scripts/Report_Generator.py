import numpy as np
import pandas as pd
from jinja2 import Environment, BaseLoader
#from jinja2 import FileSystemLoader
import webbrowser
import argparse
import os
import quantumpropagator as qp
import io
import base64
import matplotlib.pyplot as plt

def style_css():
    '''
    return style, it is mainly the format of the tables
    '''
    return '''
table.dataframe {
  font-family: "Times New Roman", Times, serif;
  border: 1px solid #FFFFFF;
  width: 350px;
  height: 200px;
  text-align: center;
  border-collapse: collapse;
}
table.dataframe td, table.dataframe th {
  border: 2px solid #FFFFFF;
  padding: 3px 2px;
}
table.dataframe tbody td {
  font-size: 13px;
}
table.dataframe tr:nth-child(even) {
  background: #D0E4F5;
}
table.dataframe thead {
  background: #0B6FA4;
  border-bottom: 5px solid #FFFFFF;
}
table.dataframe thead th {
  font-size: 17px;
  font-weight: bold;
  color: #FFFFFF;
  text-align: center;
  border-left: 2px solid #FFFFFF;
}
table.dataframe thead th:first-child {
  border-left: none;
}

table.dataframe tfoot td {
  font-size: 14px;
}
'''

def template_html():
    '''
    the html template for the report
    '''
    return '''
<!DOCTYPE html>

<html lang="en">

<head>
<style>
{{ style_string }}
</style>
    <meta http-equiv="content-type" content="text/html; charset=utf-8">
    <title> {{ title }} </title>
</head>

<body>
<h1> {{ title }} </h1>
{{ folder_string }} <br/>
report created: {{ date_string }} <br/>
{{ running_string }}
<h2> Populations </h2>
{{ popul_figure }}
<h2> Comments </h2>
{{ readme_string }}
<h2> Regions: </h2>
{{ regions_info }}
<h2> General info: </h2>
{{ info_string  }}

<h2> Norm and energies </h2>
{{ norm_figure }}
{{ kin_tot_figure }}

<h2> Raw data </h2>
{{ table_output }}

</body>

</html>
'''

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
                   'φ' : [phis_ext[-1],phis_ext[0],dphi,range_phi],
                   'γ' : [gams_ext[-1],gams_ext[0],dgam,range_gam],
                   'θ' : [thes_ext[-1],thes_ext[0],dthe,range_the]
                   }
    coordinates_df = pd.DataFrame.from_dict(little_table, orient='index')
    coordinates_df.columns = columns_df
    return (coordinates_df.to_html())

def info_pulse(dictio):
    '''
    Creates a string with a table for the pulse
    '''
    columns_df = ['E','ω','σ','φ','t0']
    little_table = {
            'X' : dictio['pulseX'],
            'Y' : dictio['pulseY'],
            'Z' : dictio['pulseZ']}
    pulse_df = pd.DataFrame.from_dict(little_table, orient='index')
    pulse_df.columns = columns_df
    return (pulse_df.to_html())

def create_string_input(dictio):
    '''
    This function transform the all input h5 file into a string of information for the report
    '''
    # 'theL', 'dphi', 'nacCube', 'nstates', 'pulseZ', 'pulseX', 'kind', 'natoms', 'kinCube', 'outFol', 'dipCube', 'phis', 'fullTime', 'dgam', 'dthe', 'h', 'gams', 'potCube', 'phiL', 'thes', 'gamL', 'pulseY'
    pres_string = 'This is a simulation of kind "{}" done in {} states<br/>dt: {:.3e} AU or {:.3e} fs<br/><br/>'
    try:
        dtAU = dictio['h']
    except KeyError:
        dtAU = dictio['dt']
    dtfs = qp.fromAuToFs(dtAU)
    pres_stringF = pres_string.format(dictio['kind'],dictio['nstates'],dtAU,dtfs)
    coord_string = '<b> Coordinates:</b>' + info_coord(dictio)
    pulse_string = '<b> Pulse specs in AU:</b>' + info_pulse(dictio)

    fullString = pres_stringF + coord_string + pulse_string
    return(fullString)

def main():
    '''
    Transform a dynamics folder into a html report
    '''

    # parse command line
    args = parseCL()

    root = os.path.dirname(os.path.abspath(args.i))
    project = os.path.basename(os.path.abspath(args.i))

    # html and style template part
    # if it is standlaone, it will use internal function for html and css file
    # otherwise it will read external files (to be used for heavy debugging/changes)
    standalone = True
    if standalone:
        style_string = style_css()
        template = Environment(loader=BaseLoader()).from_string(template_html())
    else:
        style = 'home/alessio/y-RepoQuantum/Scripts/reportGen/style.css'
        with open('style.css','r') as f:
            style_string = f.read()
        env = Environment(loader=FileSystemLoader('.'))
        template = env.get_template("report.html.j2")

    folder = os.path.join(root, project)
    allout = os.path.join(folder,'allInput.h5')
    outfn = os.path.join(folder,'output')
    outfnP = os.path.join(folder,'outputPopul')
    out_ABS = os.path.join(folder,'Output_Abs')
    readme_file = os.path.join(folder,'README')

    if os.path.isfile(readme_file):
        with open(readme_file,'r') as w:
            readme_content = w.read()
        readme_string = readme_content.replace('\n', '<br />')
    else:
        readme_string = 'No comments found in this folder'

    dictio = qp.readWholeH5toDict(allout)
    info_string = create_string_input(dictio)

    data = pd.read_csv(outfn, delim_whitespace=True, header=None);
    dataP = pd.read_csv(outfnP, delim_whitespace=True, header=None);

    # I get column number to assure there is the absorbing potential norm loss
    # March 2019, the number of column in output file is 11. This can get tricky if I change 
    # this number.
    data_col_number = data.shape[1]

    if data_col_number == 11:
        print('\n\nThis is a folder before March 2019 without absorbing potential')
        if os.path.isfile(out_ABS):
            print('An abs file is present, anyway')
            dataA = pd.read_csv(out_ABS, delim_whitespace=True, header=None);
            # I add the Abs column to this
            data = pd.concat([data, dataA], axis=1);
        else:
            qp.err('This routine now works ONLY if you have the Abs file or use last version')

    data.columns = ['count','steps','fs','Norm Deviation','Kinetic','Potential','Total','Total deviation','Xpulse','Ypulse','Zpulse','Norm Loss']
    result = pd.concat([data, dataP], axis=1);


    # title
    title_Repo = 'Report: {}'.format(project)

    #date
    import datetime
    now = datetime.datetime.now()
    date_string = now.strftime("%Y-%m-%d %H:%M")

    folder_string = folder

    # Status
    if args.running:
        running_string = 'Status: <font color="green">Simulation still running...</font>'
    else:
        running_string = 'Status: This simulation is not running/stopped'

    # first graph
    nstates = dictio['nstates']
    fig = plt.figure(figsize=(15,6))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()
    ax1.set_ylabel('Population')
    ax2.set_ylabel('Pulse')

    rename_dict = {}
    for i in range(nstates):
        rename_dict[i+1] = r"$S_{}$".format(i)

    popul = [rename_dict[i+1] for i in range(nstates)]

    result2 = result.rename(index=str, columns=rename_dict)


    colors = ['b', 'g', 'r', 'm', 'c', 'y', 'mediumpurple', 'k']

    result2.plot(title = 'Population and Pulse', ax = ax1,  x=['fs'], y=popul, linewidth=0.8, color = colors)
    result2.plot(title = 'Population and Pulse', ax = ax2,  x=['fs'], y=['Xpulse','Ypulse','Zpulse'], linewidth=0.5,ls='--', legend=False, ylim=(-0.04,0.04));

    popul_figure = fig_to_html(fig)

    # regions graph
    regions_file = '/home/alessio/n-Propagation/regions.pickle'

    if os.path.isfile(regions_file):
        import pickle
        filesList = [ fn for fn in sorted(os.listdir(folder)) if fn[:8] == 'Gaussian' and fn[-3:] == '.h5']
        if filesList != []:
            zeroWF = qp.retrieve_hdf5_data(os.path.join(folder,filesList[0]),'WF')
            phiL,gamL,theL,nstates = (qp.retrieve_hdf5_data(os.path.join(folder,filesList[0]),'WF')).shape
            filesN = len(filesList)
            allwf = np.empty((filesN,phiL,gamL,theL,nstates),dtype=complex)
            alltime = np.empty((filesN))
            for i,fn in enumerate(filesList):
                fnn = os.path.join(folder,fn)
                allwf[i] = qp.retrieve_hdf5_data(fnn,'WF')
                alltime[i] = qp.retrieve_hdf5_data(fnn,'Time')[0]
            with open(regions_file, "rb") as input_file:
                cubess = pickle.load(input_file)

            regionsN = len(cubess)

            regions_vector = np.empty((filesN,regionsN))
            fs_vector = np.empty(filesN)

            labels_region = []
            for r in range(regionsN):
                labels_region.append(cubess[r]['label'])
                for f in range(filesN):
                    if r == 0: # to do this once and not n_region times
                        time = alltime[f]
                        fs_vector[f] = time

                    uno = allwf[f,:,:,:,0] # Ground state
                    due = cubess[r]['cube']
                    value = np.linalg.norm(uno*due)
                    regions_vector[f,r] = value   # yes yes, I am swapping because of pandas

            fig_regions = plt.figure(figsize=(15,6))
            ax_regions = fig_regions.add_subplot(111)
            dataf_regions = pd.DataFrame(regions_vector, columns=labels_region)
            dataf_regions['fs'] = fs_vector
            dataf_regions.plot(title = 'S0 in different regions', ax=ax_regions, x=['fs']);
            regions_info = fig_to_html(fig_regions)
        else:
            regions_info = '<font color="red"> WARNING</font> wavefunction files not found, impossible to extract regions info.'

    else:
        regions_info = '<font color="red"> WARNING, file regions not found. Use the jupyter notebook to create one.</font>'

    # second figure
    fig2 = plt.figure(figsize=(15,4))
    ax1 = fig2.add_subplot(111)
    ax1.set_ylabel('Units')
    data.plot(title = 'Norm Deviation', ax=ax1, x='fs', y = 'Norm Deviation')
    norm_figure = fig_to_html(fig2)


    # third figure
    fig3 = plt.figure(figsize=(15,4))
    ax1 = fig3.add_subplot(111)
    ax1.set_ylabel('Ev')
    data['Kinetic_Moved'] = data['Kinetic'] + data['Potential'][0]
    data.plot(title = 'Comparison Potential Total Kinetic', ax=ax1, x=['fs'] ,y=['Kinetic_Moved','Potential','Total'], figsize=(15,5))

    kin_tot_figure = fig_to_html(fig3)

    df2 = pd.DataFrame(result)

    # setting the html
    template_vars = {"title" : title_Repo,
                     "table_output": df2.to_html(max_rows=50),
                     "folder_string" : folder_string,
                     "date_string" : date_string,
                     "running_string": running_string,
                     "regions_info" : regions_info,
                     "info_string": info_string,
                     "popul_figure": popul_figure,
                     "kin_tot_figure": kin_tot_figure,
                     "readme_string" : readme_string,
                     "style_string": style_string,
                     "norm_figure": norm_figure}

    html_out = template.render(template_vars)

    filename = 'Report_{}.html'.format(project)

    with open(filename, 'w') as f:
        f.write(html_out)

    print('\nFile {} written.\n'.format(filename))

    if args.data:
        filename_popu = 'Report_{}_populations.csv'.format(project)
        filename_region = 'Report_{}_regions.csv'.format(project)
        df2.to_csv(filename_popu)
        dataf_regions.to_csv(filename_region)

    # open the browser or not
    if args.browser:
        webbrowser.open(filename)

def parseCL():
    d = 'This tools is used to generate html reports'
    parser = argparse.ArgumentParser(description=d)
    parser.add_argument("-i", "--input",
                        dest="i",
                        required=True,
                        type=str,
                        help="path of the folder for the analysis")
    parser.add_argument("-f", "--firefox",
                        dest="browser",
                        action='store_true',
                        help="launches the browser")
    parser.add_argument("-r", "--running",
                        dest="running",
                        action='store_true',
                        help="tells the report that the calculation is running")
    parser.add_argument("-d", "--data-raw",
                        dest="data",
                        action='store_true',
                        help="creates a csv raw file alongside the html")

    return parser.parse_args()



if __name__ == "__main__":
    main()
