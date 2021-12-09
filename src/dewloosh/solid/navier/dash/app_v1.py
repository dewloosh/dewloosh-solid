# -*- coding: utf-8 -*-
from dewloosh.solid.navier.dash import calc3d, fig2d, fig3d, \
    input_mat, input_geom, input_calc, input_load
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State


dofs = UZ, ROTX, ROTY, CX, CY, CXY, EXZ, EYZ, MX, MY, MXY, QX, QY = list(range(13))
id_to_label = {UZ : 'UZ', ROTX : 'ROTX', ROTY : 'ROTY', CX : 'CX', 
               CY : 'CY', CXY : 'CXY', EXZ : 'EXZ', EYZ : 'EYZ', 
               MX : 'MX', MY : 'MY', MXY : 'MXY', QX : 'QX', QY : 'QY'}
label_to_id = {value:key for key, value in id_to_label.items()}
proj = '3d'


# inital parameters and plotting
Lx, Ly = 600., 800.
nx, ny = 50, 50
rx, ry = 30, 40
E = 2890.
nu = 0.2
t = 25.
x0, y0, w, h, q = 0.2*Lx, 0.5*Ly, 0.2*Lx, 0.3*Ly, -0.1
params = {'E' : E, 'nu' : nu, 't' : t, 'Lx' : Lx, 'Ly' : Ly, 
          'x0' : x0, 'y0' : y0, 'w' : w, 'h' : h, 'q' : q,
          'nx' : nx, 'ny' : ny, 'rx' : rx, 'ry' : ry}
coords, triangles, res2d  = calc3d(**params)
if proj == '3d':
    fig = fig3d(coords, triangles, res2d[0, UZ, :], **params)
elif proj == '2d':
    fig = fig2d(coords, res2d[0, UZ, :], **params)


app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.layout = dbc.Container(
    [
        html.H1(children='Mindlin vs. Kirchhoff'),
        html.P(
                "A simple dashboard to compare Mindlin-Reissner" \
                    + " and Kirchhoff-Love plates.",
                className="lead",
                ),
        html.Hr(),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2(children='Geometry'),
                        input_geom(**params),
                        
                        html.H2(children='Material'),
                        input_mat(**params),
                 
                        html.H2(children='Load'),
                        input_load(**params),
                        
                        html.H2(children='Calculation'),
                        input_calc(**params),
                        
                        dbc.Button(
                            "Calculate", 
                            id='calc_button', 
                            color="primary"
                            )
                            
                    ], width=3
                ),
                
                dbc.Col(
                    [
                        dbc.Row([
                            
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id='component',
                                        options = [{'label' : id_to_label[dof], 
                                                    'value' : id_to_label[dof]} 
                                                for dof in dofs],
                                        value='UZ'
                                    ),
                                ]
                            ),
                            
                            dbc.Col(
                                [
                                    dcc.Dropdown(
                                        id='model',
                                        options = [
                                            {'label' : 'Mindlin', 'value' : 'mindlin'},
                                            {'label' : 'Kirchhoff', 'value' : 'kirchhoff'}
                                            ],
                                        value='mindlin'
                                    ),
                                ]
                            ),
                            
                        ]),
                        
                        dcc.Graph(id='plot', figure=fig),  
                        
                    ], width=9
                ),
            ]
        )
    ],
    fluid=True,
)


@app.callback(
    Output('plot', 'figure'),
    Input('component', 'value'),
    Input('model', 'value'))
def update_figure(comp, model):
    global res2d, params, coords, triangles
    dId = label_to_id[comp]
    cmap = "Viridis" if model == 'mindlin' else "Bluered"
    mId = 0 if model == 'mindlin' else 1
    if proj == '3d':
        fig = fig3d(coords, triangles, res2d[mId, dId, :], 
                    cmap=cmap, **params)
        
    elif proj == '2d':
        fig = fig2d(coords, res2d[mId, dId, :], cmap=cmap, **params)
    return fig


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
    State('component', 'value'),
    )
def recalc(n_clicks, E, nu, t, Lx, Ly, x0, y0, w, h, q, nx, ny, comp):
    global res2d, coords, triangles
    params = {'E' : E, 'nu' : nu, 't' : t, 'Lx' : Lx, 'Ly' : Ly, 
              'x0' : x0, 'y0' : y0, 'w' : w, 'h' : h, 'q' : q}
    params = {key : float(value) for key, value in params.items()}
    params['nx'] = int(nx)
    params['ny'] = int(ny)
    coords, triangles, res2d = calc3d(**params)
    return comp


if __name__ == '__main__':
    app.run_server(debug=True)