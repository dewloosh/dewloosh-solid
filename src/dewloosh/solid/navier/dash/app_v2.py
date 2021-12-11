# -*- coding: utf-8 -*-
from dewloosh.core.tools import float_to_str_sig
from dewloosh.solid.navier.dash import solve, fig2d, fig3d, \
    input_mat, input_geom, input_calc, input_load, \
    gen_problem, gen_grid, input_res
import dash_bootstrap_components as dbc
from dash import Dash as DashBoard, dcc, html, dash_table as dt
from dash.dependencies import Input, Output, State
import pandas as pd
import numpy as np


def pprint(x): return float_to_str_sig(x, sig=6, atol=1e-10)


dofs = UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY = list(
    range(13))
id_to_label = {UZ: 'UZ', ROTX: 'ROTX', ROTY: 'ROTY', CX: 'CX',
               CY: 'CY', CXY: 'CXY', EXZ: 'EXZ', EYZ: 'EYZ',
               MX: 'MX', MY: 'MY', MXY: 'MXY', QX: 'QX', QY: 'QY'}
label_to_id = {value: key for key, value in id_to_label.items()}
proj = '2d'

# inital parameters and plotting
Lx, Ly = 800., 400.
nx, ny = 20, 20
rx, ry = 30, 30
E = 2890.
nu = 0.2
t = 25.
x0, y0, w, h, q = 100., 100., 100., 100., -0.1
params = {'E': E, 'nu': nu, 't': t, 'Lx': Lx, 'Ly': Ly,
          'x0': x0, 'y0': y0, 'w': w, 'h': h, 'q': q,
          'nx': nx, 'ny': ny, 'rx': rx, 'ry': ry}
tbldata = [
    ['min', 0, 0, 0],
    ['max', 0, 0, 0],
]

# calculation
Navier = gen_problem(**params)
if proj == '3d':
    Mesh = gen_grid(proj=proj, **params)
    coords = Mesh.centers()
    triangles = Mesh.topology()
    res2d = solve(Navier, coords)
    coords = Mesh.coords()
elif proj == '2d':
    coords = gen_grid(proj=proj, **params)
    res2d = solve(Navier, coords)

# table
tbldata[0][1] = pprint(res2d[0, UZ, :].min())
tbldata[1][1] = pprint(res2d[0, UZ, :].max())
tbldata[0][2] = pprint(res2d[1, UZ, :].min())
tbldata[1][2] = pprint(res2d[1, UZ, :].max())
for i in range(2):
    MK = np.array(tbldata[i][1:3]).astype(float)
    if np.min(np.abs(MK)) < 1e-12:
        tbldata[i][3] = "0 %"
    else:
        try:
            fstr = pprint(100*(MK[0] - MK[1]) / np.abs(MK[1]))
            tbldata[i][3] = "{} %".format(fstr)
        except Exception:
            tbldata[i][3] = 'nan'
tbldf = pd.DataFrame(
    tbldata,
    columns=['', 'Mindlin', 'Kirchhoff', 'K --> M']
)

# figure
if proj == '3d':
    fig = fig3d(coords, triangles, res2d[0, :], **params)
elif proj == '2d':
    fig = fig2d(coords, res2d[0, :], **params)

SIDEBAR_STYLE = {
    "background-color": "#f8f9fa",
}
navigation_panel = html.Div(
    [
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        input_geom(**params),
                    ],
                    title="Geometry",
                ),
                dbc.AccordionItem(
                    [
                        input_mat(**params),
                    ],
                    title="Material",
                ),
                dbc.AccordionItem(
                    [
                        input_load(**params),
                    ],
                    title="Load",
                ),
                dbc.AccordionItem(
                    [
                        input_calc(**params),
                    ],
                    title="Calculation",
                ),
                dbc.AccordionItem(
                    [
                        input_res(**params)
                    ],
                    title="Results",
                ),
            ],
        ),
        html.Br(),
        dbc.Button(
            "Calculate",
            id='calc_button',
            color="primary"
        )
    ],
    style=SIDEBAR_STYLE,
)


# DASHBOARD
app = DashBoard(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    dbc.Row([
        dbc.Col(
            [
                html.H1(children='Mindlin vs. Kirchhoff'),
                html.P(
                    "An AxisVM dashboard to compare Mindlin-Reissner"
                    + " and Kirchhoff-Love plates.",
                    className="lead",
                ),
                navigation_panel
            ],
            width=3
        ),

        dbc.Col(
            [
                dcc.Graph(id='plot', figure=fig),
                dt.DataTable(
                    id='tbl', data=tbldf.to_dict('records'),
                    columns=[{"name": i, "id": i} for i in tbldf.columns],
                ),
            ],
            width=9
        ),
    ]),
    fluid=True,
)


@app.callback(
    Output('plot', 'figure'),
    Output('tbl', 'data'),
    Input('component', 'value'),
    Input('model', 'value'))
def update_figure(comp, model):
    global res2d, tbldata, params, coords, triangles
    dId = label_to_id[comp]
    cmap = "Viridis" if model == 'mindlin' else "Bluered"
    mId = 0 if model == 'mindlin' else 1
    # figure
    if proj == '3d':
        fig = fig3d(coords, triangles, res2d[mId, dId, :],
                    cmap=cmap, **params)
    elif proj == '2d':
        fig = fig2d(coords, res2d[mId, dId, :], cmap=cmap, **params)
    # table
    tbldata[0][1] = pprint(res2d[0, dId, :].min())
    tbldata[1][1] = pprint(res2d[0, dId, :].max())
    tbldata[0][2] = pprint(res2d[1, dId, :].min())
    tbldata[1][2] = pprint(res2d[1, dId, :].max())
    for i in range(2):
        MK = np.array(tbldata[i][1:3]).astype(float)
        if np.min(np.abs(MK)) < 1e-12:
            tbldata[i][3] = "0 %"
        else:
            try:
                fstr = pprint(100*(MK[0] - MK[1]) / np.abs(MK[1]))
                tbldata[i][3] = "{} %".format(fstr)
            except Exception:
                tbldata[i][3] = 'nan'
    tbldf = pd.DataFrame(
        tbldata,
        columns=['', 'Mindlin', 'Kirchhoff', 'K --> M']
    )
    return fig, tbldf.to_dict('records')


@app.callback(
    Output('component', 'value'),
    Input('calc_button', 'n_clicks'),
    State('E', 'value'),
    State('nu', 'value'),
    State('t', 'value'),
    State('Lx', 'value'),
    State('Ly', 'value'),
    State('x0', 'value'),
    State('y0', 'value'),
    State('w', 'value'),
    State('h', 'value'),
    State('q', 'value'),
    State('nx', 'value'),
    State('ny', 'value'),
    State('rx', 'value'),
    State('ry', 'value'),
    State('component', 'value'),
)
def recalc(n_clicks, E, nu, t, Lx, Ly, x0, y0, w, h, q,
           nx, ny, rx, ry, comp):
    global res2d, coords, triangles, Navier, pointdata, params
    params = {'E': E, 'nu': nu, 't': t, 'Lx': Lx, 'Ly': Ly,
              'x0': x0, 'y0': y0, 'w': w, 'h': h, 'q': q}
    params = {key: float(value) for key, value in params.items()}
    params['nx'] = int(nx)
    params['ny'] = int(ny)
    params['rx'] = int(rx)
    params['ry'] = int(ry)
    Navier = gen_problem(**params)

    if proj == '3d':
        Mesh = gen_grid(proj=proj, **params)
        coords = Mesh.centers()
        triangles = Mesh.topology()
        res2d = solve(Navier, coords)
        coords = Mesh.coords()
    elif proj == '2d':
        coords = gen_grid(proj=proj, **params)
        res2d = solve(Navier, coords)
    return comp


if __name__ == '__main__':
    app.run_server(debug=True)
